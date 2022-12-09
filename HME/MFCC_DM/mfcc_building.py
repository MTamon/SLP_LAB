from logging import Logger
from argparse import Namespace
from typing import Dict, List, Tuple
from tqdm import tqdm
from multiprocessing import Pool

from pydub import AudioSegment
import wave
import os
import re
import pickle
import math
import numpy as np

from feature_extractor import FeatureExtractor
from logger_gen import set_logger
from utils import batching, get_mfcc_args
from modules import Collector, Condition
from modules import load_index_file, load_luu_csv, load_shaped
from modules import tools


def wav_condition(path: str) -> bool:
    """For CEJC dataset's condition"""

    dir_path = os.path.dirname(path)
    fil_name = os.path.basename(path)
    member_list = os.listdir(dir_path)

    if fil_name.split("_")[-1] == "IC0B.wav":
        return True

    if fil_name.replace("IC0A", "IC0B") in member_list:
        return False
    else:
        return True


def group_key(fname: str) -> str:
    ftype = re.split(r"_.", fname)[1]
    _ftype = ftype[:3] + "#"

    if len(ftype) > 4:
        _ftype += ftype[4:]

    return _ftype


class Mfcc_Segment:
    def __init__(self, args: Namespace, logger: Logger = None) -> None:
        if logger is None:
            self.logger = set_logger("MFCC-SEG", args.log)
        else:
            self.logger = logger

        self.args = args

        self.target = args.target
        self.output = args.output

        self.segment_size = args.segment_size
        self.segment_stride = args.segment_stride
        self.segment_min_size = args.segment_min_size

        self.sep_data = args.sep_data
        self.feature = args.used_feature

        self.video_fps = args.video_fps

        self.sample_frequency = args.sample_frequency
        self.frame_length = args.frame_length
        self.frame_shift = args.frame_shift
        self.num_mel_bins = args.num_mel_bins
        self.num_ceps = args.num_ceps
        self.low_frequency = args.low_frequency
        self.high_frequency = args.high_frequency
        self.dither = args.dither

        self.proc_num = args.proc_num
        self.single_proc = args.single_proc

        if self.high_frequency * 2 > self.sample_frequency:
            raise ValueError(
                f"High-frequency must be smaller than half of sample-frequency. \
                But high={self.high_frequency}, sample={self.sample_frequency}"
            )

        self.feat_extractor = FeatureExtractor(
            sample_frequency=args.sample_frequency,
            frame_length=args.frame_length,
            frame_shift=args.frame_shift,
            point=True,
            num_mel_bins=args.num_mel_bins,
            num_ceps=args.num_ceps,
            low_frequency=args.low_frequency,
            high_frequency=args.high_frequency,
            dither=args.dither,
        )

        self.temp_path = "/".join(re.split(r"[\\]", self.output).append("tmp"))
        if not os.path.isdir(self.temp_path):
            os.mkdir(self.temp_path)
        self.seg_path = "/".join(re.split(r"[\\]", self.output).append("data"))
        if not os.path.isdir(self.seg_path):
            os.mkdir(self.seg_path)

        condition1 = Condition().specify_extention(["avidx"])
        condition1 = condition1.add_exclude_dirc(["DONOT_USE"])

        condition2 = Condition().specify_extention(["wav"])
        condition2 = condition2.add_exclude_dirc(["DONOT_USE"])
        condition2 = condition2.add_contain_filename(["IC0A", "IC0B"])
        condition2 = condition2.add_condition_func(wav_condition)

        self.conditions = (condition1, condition2)

        self.collector1 = Collector(self.conditions, self.target)
        self.database = self.collector1.get_directory_instance().clone(self.conditions)

        self.idx_ic0a = self.get_file_groups()

        if self.segment_size < self.segment_min_size:
            raise ValueError(
                "--segment-size must be smaller than --segment-min-size. But {0} & {1}".format(
                    self.segment_size, self.segment_min_size
                )
            )
        if not self.feature in ["mfcc", "fbank"]:
            raise ValueError(
                "--used-feature must be 'mfcc' or 'fbank'. But {0}".format(self.feature)
            )

    def __call__(self) -> Tuple[List[str], float, int]:

        idx_set = []
        for i, dt in enumerate(self.idx_ic0a.items()):
            idx_set.append([i, *dt])

        batches = batching(idx_set, batch_size=self.proc_num)
        iterator = tqdm(batches, desc=" load .avidx :")

        results = []

        for batch in iterator:
            if not self.single_proc:
                with Pool(processes=None) as pool:
                    results += pool.starmap(self.phase, batch)
            else:
                for _ba in batch:
                    results.append(self.phase(*_ba))

        _results = []
        time_all = 0
        frame_all = 0
        for result in results:
            if result[0] is None:
                self.logger.info("! Reject data > %s: %s ... %s (%s)", *result[1])
            else:
                _results.append(result[0])
                time_all += result[2]["time"]
                frame_all += result[2]["frame"]

        return (_results, time_all, frame_all)

    def phase(self, file_idx, _avidx, _ic0a) -> List[Tuple[str, tuple]]:
        result = []

        avidx = load_index_file(_avidx)
        for pair in avidx["pairs"]:
            wpath = pair["wav"]
            spkID = pair["spID"]
            csv_dt = load_luu_csv(avidx["csv"])
            shp_dt = load_shaped(pair["sh"])

            fps = shp_dt[3]
            shp_dt = shp_dt[0]

            _result = self.check_files(pair, fps, _ic0a)
            if _result != []:
                result += _result
                continue

            video_stride = math.ceil(fps * self.segment_stride)
            video_min_size = math.ceil(fps * self.segment_min_size)
            video_size = math.ceil(fps * self.segment_size)

            _stride_rest = 0
            segment_id = 0

            segment = self.create_segment_dict(fps)
            for start, shp_rec in enumerate(shp_dt):
                if _stride_rest > 0:
                    _stride_rest -= 1
                    continue
                if shp_rec["ignore"]:
                    continue

                for nframe in range(video_size + 1):
                    # finish condition
                    if shp_rec["ignore"] or nframe == video_size:
                        if nframe < video_min_size:
                            segment = self.create_segment_dict(fps)
                            break

                        term = np.array([start, start + nframe]) / fps

                        segment["cent"] = np.stack(segment["cent"])
                        segment["angl"] = np.stack(segment["angl"])
                        segment["trgt"] = self.get_feature(wpath, *term, csv_dt, spkID)
                        segment["othr"] = self.get_feature(_ic0a, *term, csv_dt, spkID)
                        segment["ffps"] = segment["trgt"] / (term[1] - term[0])
                        segment["term"] = term - term[0]

                        _name = "_".join(["data", file_idx, segment_id]) + ".seg"
                        _segment_path = "/".join([self.seg_path, _name])
                        self.write_segment(segment, _segment_path)

                        _info = {"time": term[1] - term[0], "frame": nframe}
                        result.append((_segment_path, None, _info))

                        segment = self.create_segment_dict(fps)

                        _stride_rest = video_stride - 1
                        break

                    centroid = shp_dt["centroid"]
                    euler = tools.rotation_angles(shp_dt["rotate"].T)

                    segment["cent"].append(centroid)
                    segment["angl"].append(euler)

    def get_feature(self, wav_path, start, stop, csv_dt, spID) -> np.ndarray:

        _start = int(self.sample_frequency * start)
        _stop = int(self.sample_frequency * stop)

        name = os.path.basename(wav_path)

        np.random.seed(seed=0)

        with wave.open(wav_path, mode="r") as wav:
            if wav.getnchannels() == 2:
                _wav_path = "/".join([self.temp_path, name])
                if not os.path.isfile(_wav_path):
                    sound = AudioSegment.from_wav(wav_path)
                    sound = sound.set_channels(1)
                    sound.export(_wav_path, format="wav")
                wav_path = _wav_path

        with wave.open(wav_path, mode="r") as wav:

            _num_samples = wav.getnframes()

            # read as binary & convert np.int16
            _waveform = wav.readframes(_num_samples)
            _waveform = np.frombuffer(_waveform, dtype=np.int16)

            segment_wav = _waveform[_start:_stop]

            if self.feature == "mfcc":
                _feature = self.feat_extractor.ComputeMFCC(segment_wav)
                assert _feature.shape[1] == self.num_ceps
            elif self.feature == "fbank":
                _feature, _log_pow = self.feat_extractor.ComputeFBANK(segment_wav)
                assert _feature.shape[1] == self.num_mel_bins

        _feature = _feature.astype(np.float32)

        if self.sep_data:
            w_name = os.path.basename(wav_path)
            mask = np.zeros(len(_feature))
            ffps = len(_feature) / (stop - start)

            for _rec in csv_dt:
                if _rec["speakerID"] != spID:
                    continue

                strt_time = _rec["startTime"]
                stop_time = _rec["endTime"]
                if not (start <= strt_time <= stop or start <= stop_time <= stop_time):
                    continue

                luu_strt = max(start, strt_time) - start
                luu_stop = min(stop, stop_time) - start
                strt_frame = luu_strt * ffps
                stop_frame = luu_stop * ffps

                mask[strt_frame:stop_frame] = 1

            if spID in w_name:
                _feature = mask * _feature
            else:
                _feature = (1 - mask) * _feature

        return _feature

    def check_files(self, pair, fps, _ic0a) -> List[Tuple[str, tuple]]:
        result = []
        if fps != self.video_fps:
            name = os.path.basename(pair["sh"])
            log_info = ("fps", name, fps, self.video_fps)
            result.append((None, log_info, None))

        with wave.open(_ic0a, mode="r") as wf:
            sf = wf.getframerate()
            if sf != self.sample_frequency:
                name = os.path.basename(_ic0a)
                log_info = ("sample frequency", name, fps, self.sample_frequency)
                result.append((None, log_info, None))

            ch = wf.getnchannels()
            if ch != 2:
                name = os.path.basename(_ic0a)
                log_info = ("channel", name, fps, 2)
                result.append((None, log_info, None))

        wav_path = pair["wav"]
        with wave.open(wav_path, mode="r") as wf:
            sf = wf.getframerate()
            if sf != self.sample_frequency:
                name = os.path.basename(wav_path)
                log_info = ("sample frequency", name, fps, self.sample_frequency)
                result.append((None, log_info, None))

            if ch != 1:
                name = os.path.basename(wav_path)
                log_info = ("channel", name, fps, 1)
                result.append((None, log_info, None))

        return result

    def create_segment_dict(self, fps) -> dict:
        segment = {
            "cent": [],
            "angl": [],
            "trgt": [],
            "othr": [],
            "vfps": fps,
            "afps": self.sample_frequency,
            "ffps": 0,
            "term": 0,
        }
        return segment

    def get_file_groups(self) -> Dict[str, str]:
        _idx_ic0a = self.database.get_terminal_instances()
        _idx_ic0a = [dirc.get_grouped_path_list(group_key) for dirc in _idx_ic0a]
        idx_ic0a = []
        for _r in _idx_ic0a:
            assert len(_r) == 2
            idx_ic0a += _r

        _d = {}
        for _r in idx_ic0a:
            if _r[0].split(".")[-1] == "avidx":
                _d[_r[0]] = _r[1]
            else:
                _d[_r[1]] = _r[0]

        return _d

    def write_segment(self, segment, path):
        # output by pickle
        with open(path, "wb") as f:
            pickle.dump(segment, f)


if __name__ == "__main__":
    _args = get_mfcc_args()
    _logger = set_logger(_args.log)
    segmenter = Mfcc_Segment(_args, _logger)

    path_list, _time_all, _frame_all = segmenter()

    for _path in path_list:
        _logger.info("Create segment : %s", _path)
    _logger.info("All time  : %s[s]", _time_all)
    _logger.info("All frame : %s", _frame_all)
