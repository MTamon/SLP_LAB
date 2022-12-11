from argparse import Namespace
from logging import Logger
from multiprocessing import Pool
from typing import Dict, List, Tuple
import math
import os
import pickle
import re
import wave
import sox
from tqdm import tqdm
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
    ftype = re.split(r"[_.]", fname)[1]
    _ftype = ftype[:3] + "#"

    if len(ftype) > 3:
        _ftype = ftype

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
        self.feature = args.use_feature

        self.video_fps = args.video_fps

        self.sample_frequency = args.sample_frequency
        self.frame_length = args.frame_length
        self.frame_shift = args.frame_shift
        self.num_mel_bins = args.num_mel_bins
        self.num_ceps = args.num_ceps
        self.low_frequency = args.low_frequency
        self.high_frequency = args.high_frequency
        self.dither = args.dither

        self.point_size = int(self.sample_frequency * self.frame_length * 0.001)
        self.point_shift = int(self.sample_frequency * self.frame_shift * 0.001)
        self.feature_width_min = (
            int(self.segment_min_size * self.sample_frequency - self.point_size)
            // self.point_shift
        )

        self.proc_num = args.proc_num
        self.single_proc = args.single_proc

        self.convert = args.convert_path
        self.redo = args.redo

        if self.high_frequency * 2 > self.sample_frequency:
            raise ValueError(
                f"High-frequency must be smaller than half of sample-frequency. \
                But high={self.high_frequency}, sample={self.sample_frequency}"
            )

        self.feat_extractor = FeatureExtractor(
            sample_frequency=args.sample_frequency,
            frame_length=args.frame_length,
            frame_shift=args.frame_shift,
            # point=True,
            num_mel_bins=args.num_mel_bins,
            num_ceps=args.num_ceps,
            low_frequency=args.low_frequency,
            high_frequency=args.high_frequency,
            dither=args.dither,
        )

        self.output = "/".join(re.split(r"[\\]", self.output))
        if not os.path.isdir(self.output):
            os.mkdir(self.output)
        self.temp_path = self.output + "/tmp"
        if not os.path.isdir(self.temp_path):
            os.mkdir(self.temp_path)
        self.seg_path = self.output + "/data"
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
        for (_avidx, _ic0a) in self.idx_ic0a.items():
            avidx = load_index_file(_avidx)
            avidx["csv"] = self.convert_path(avidx["csv"])
            avidx["name"] = self.convert_path(avidx["name"])

            for pair in avidx["pairs"]:

                pair["wav"] = self.convert_path(pair["wav"])
                pair["sh"] = self.convert_path(pair["sh"])

                name = os.path.basename(_ic0a)

                _wav_path = "/".join([self.temp_path, name])
                if not os.path.isfile(_wav_path):
                    with wave.open(_ic0a, mode="r") as wav:
                        if wav.getnchannels() == 2:
                            trm = sox.Transformer()
                            trm.convert(n_channels=1)
                            trm.build(_ic0a, _wav_path)
                            _ic0a = _wav_path
                            self.logger.info(" Convert to monaural audio from %s", name)
                else:
                    _ic0a = _wav_path

                idx_set.append([avidx, _ic0a, pair, False])

        if self.single_proc:
            batches = idx_set
        else:
            batches = batching(idx_set, batch_size=self.proc_num)

        results = []

        for progres, batch in enumerate(batches):
            self.logger.info(" >> Progress %s/%s", progres + 1, len(batches))
            if not self.single_proc:
                batch[0][-1] = True
                with Pool(processes=None) as pool:
                    results += pool.starmap(self.phase, batch)
            else:
                batch[-1] = True
                results.append(self.phase(*batch))

        _results = []
        time_all = 0
        frame_all = 0
        for result_group in results:
            for result in result_group:
                if result[0] is None:
                    self.logger.info("! Reject data > %s: %s ... %s (%s)", *result[1])
                else:
                    _results.append(result[0])
                    time_all += result[2]["time"]
                    frame_all += result[2]["frame"]

        return (_results, time_all, frame_all)

    def phase(self, avidx, _ic0a, pair, tqdm_visual=False) -> List[Tuple[str, tuple]]:
        result = []

        f_name = "_".join(os.path.basename(pair["sh"]).split("_")[:4])
        name = " " * (17 - len(f_name)) + f_name + " "

        wpath = pair["wav"]
        spkID = pair["spID"]
        csv_dt = load_luu_csv(avidx["csv"])
        shp_dt = load_shaped(pair["sh"])

        fps = shp_dt[3]
        shp_dt = shp_dt[0]

        _result = self.check_files(pair, fps, _ic0a)
        if _result:
            if tqdm_visual:
                self.logger.info(" >> ! Reject : %s", name)
            return _result

        _csv_dt = []
        for _rec in csv_dt:
            if _rec["speakerID"].split("_")[0] == spkID:
                _csv_dt.append(_rec)
        csv_dt = sorted(_csv_dt, key=lambda x: x.get("startTime", 1e5), reverse=False)

        video_stride = math.ceil(fps * self.segment_stride)
        video_min_size = math.ceil(fps * self.segment_min_size)
        video_size = math.ceil(fps * self.segment_size)

        _stride_rest = 0

        segment = self.create_segment_dict(fps)

        if tqdm_visual:
            shp_iterator = tqdm(shp_dt, desc=name)
        else:
            shp_iterator = shp_dt

        for start, shp_rec in enumerate(shp_iterator):
            _csv_dt = csv_dt.copy()
            for _rec in _csv_dt:
                if _rec["endTime"] < start / fps:
                    csv_dt.pop(0)
                else:
                    break
            del _csv_dt

            if _stride_rest > 0:
                _stride_rest -= 1
                continue
            if shp_rec["ignore"]:
                continue

            for nframe in range(video_size + 1):
                current_idx = start + nframe

                cond1 = nframe == video_size
                cond2 = current_idx >= len(shp_dt)
                cond3 = False
                if not cond2:
                    cond3 = shp_dt[current_idx]["ignore"]

                # finish condition
                if cond1 or cond2 or cond3:
                    if nframe - 1 < video_min_size:
                        segment = self.create_segment_dict(fps)
                        _stride_rest = nframe - 1
                        break

                    _term = np.array([start, current_idx - 1])
                    term = _term / fps

                    _stride_rest = video_stride - 1
                    _info = {"time": term[1] - term[0], "frame": nframe}

                    _name = "_".join([f_name, str(_term[0]), str(_term[1])]) + ".seg"
                    _segment_path = "/".join([self.seg_path, _name])
                    if os.path.isfile(_segment_path) and not self.redo:
                        segment = self.create_segment_dict(fps)
                        result.append((_segment_path, None, _info))
                        break

                    segment["cent"] = np.stack(segment["cent"])
                    segment["angl"] = np.stack(segment["angl"])
                    segment["trgt"] = self.get_feature(wpath, *term, csv_dt, spkID)
                    segment["othr"] = self.get_feature(_ic0a, *term, csv_dt, spkID)
                    segment["ffps"] = len(segment["trgt"]) / (term[1] - term[0])
                    segment["term"] = term - term[0]

                    assert segment["trgt"].shape == segment["othr"].shape

                    result.append((_segment_path, None, _info))

                    self.write_segment(segment, _segment_path)
                    segment = self.create_segment_dict(fps)
                    break

                centroid = shp_dt[current_idx]["centroid"]
                euler = tools.rotation_angles(shp_dt[current_idx]["rotate"].T)

                segment["cent"].append(centroid)
                segment["angl"].append(euler)

        return result

    def get_feature(self, wav_path, start, stop, csv_dt, spID) -> np.ndarray:
        assert (
            stop - start > self.segment_min_size
        ), f"term {start} - {stop} (min {self.segment_min_size})"

        _start = int(self.sample_frequency * start)
        _stop = int(self.sample_frequency * stop)

        np.random.seed(seed=0)

        with wave.open(wav_path, mode="r") as wav:

            _num_samples = wav.getnframes()

            # read as binary & convert np.int16
            _waveform = wav.readframes(_num_samples)
            _waveform = np.frombuffer(_waveform, dtype=np.int16)

            segment_wav = _waveform[_start:_stop]
            assert _stop - 1 < len(_waveform), "Over time: wave {0}, endTerm{1}".format(
                len(_waveform), _stop
            )

            mask = self.generate_mask(
                len(segment_wav), start, stop, spID, wav_path, csv_dt
            )
            if np.sum(mask) == 0:
                return np.zeros((len(mask), self.num_mel_bins))

            if self.feature == "mfcc":
                _feature = self.feat_extractor.ComputeMFCC(segment_wav)
                assert _feature.shape[1] == self.num_ceps
            elif self.feature == "fbank":
                _feature, _log_pow = self.feat_extractor.ComputeFBANK(segment_wav)
                assert _feature.shape[1] == self.num_mel_bins

        _feature = _feature.astype(np.float32)

        if self.sep_data:
            _feature = mask.reshape((-1, 1)) * _feature

        return _feature

    def generate_mask(
        self, num_samples, start, stop, spID, wav_path, csv_dt
    ) -> np.ndarray:

        flength = (num_samples - self.point_size) // self.point_shift + 1
        assert (
            flength > self.feature_width_min
        ), f"flength {flength}, feature-width {self.feature_width_min}"

        mask = np.zeros(flength)
        ffps = flength / (stop - start)

        w_name = os.path.basename(wav_path)

        for _rec in csv_dt:
            strt_time = _rec["startTime"]
            stop_time = _rec["endTime"]
            if strt_time > stop:
                break
            if not (start <= strt_time <= stop or start <= stop_time <= stop):
                if not (strt_time <= start and stop <= stop_time):
                    continue

            if _rec["speakerID"].split("_")[0] != spID:
                continue

            luu_strt = max(start, strt_time) - start
            luu_stop = min(stop, stop_time) - start
            strt_frame = round(luu_strt * ffps)
            stop_frame = round(luu_stop * ffps)

            mask[strt_frame:stop_frame] = 1

        if spID in w_name:
            return mask
        else:
            return 1 - mask

    def check_files(self, pair, fps, _ic0a) -> List[Tuple[str, tuple]]:
        def check_fps() -> bool:
            _fps = str(self.video_fps).split(".")

            if len(_fps) == 1:
                _fps = ""
            else:
                _fps = _fps[1]

            return round(fps, len(_fps)) == self.video_fps

        result = []
        if not check_fps():
            name = os.path.basename(pair["sh"])
            log_info = ("fps", name, fps, self.video_fps)
            result.append((None, log_info, None))

        for wav_path in [pair["wav"], _ic0a]:
            with wave.open(wav_path, mode="r") as wf:
                sf = wf.getframerate()
                if sf != self.sample_frequency:
                    name = os.path.basename(wav_path)
                    log_info = ("sample frequency", name, sf, self.sample_frequency)
                    result.append((None, log_info, None))

                ch = wf.getnchannels()
                if ch != 1:
                    name = os.path.basename(wav_path)
                    log_info = ("channel", name, ch, 1)
                    result.append((None, log_info, None))

        return result

    def create_segment_dict(self, fps) -> dict:
        segment = {
            "cent": [],
            "angl": [],
            "trgt": None,
            "othr": None,
            "vfps": fps,
            "afps": self.sample_frequency,
            "ffps": 0,
            "term": 0,
        }
        return segment

    def get_file_groups(self) -> Dict[str, str]:
        _idx_ic0a = self.database.get_terminal_instances(serialize=True)
        _idx_ic0a = [dirc.get_grouped_path_list(group_key) for dirc in _idx_ic0a]

        idx_ic0a = []
        for _grouped in _idx_ic0a:
            for _group in _grouped:
                assert len(_group) in [2, 1], f"{_group}\nsize {len(_group)}"
                if len(_group) == 1:
                    assert (
                        _group[0].split(".")[-1] == "wav"
                    ), f"Invalid file : {os.path.basename(_group[0])}"
                else:
                    idx_ic0a += [_group]

        _d = {}
        for _r in idx_ic0a:
            if _r[0].split(".")[-1] == "avidx":
                _d[_r[0]] = _r[1]
            else:
                _d[_r[1]] = _r[0]

        return _d

    def convert_path(self, path):
        if self.convert is None:
            return path

        r_path = re.split(r"[\\/]", path)[self.convert :]
        _target = re.split(r"[\\/]", self.target)
        path = "/".join(_target) + "/" + "/".join(r_path)

        return path

    def write_segment(self, segment, path):
        # output by pickle
        with open(path, "wb") as f:
            pickle.dump(segment, f)


if __name__ == "__main__":
    _args = get_mfcc_args()
    _logger = set_logger("SEGMENT", _args.log)
    segmenter = Mfcc_Segment(_args, _logger)

    path_list, _time_all, _frame_all = segmenter()

    for _path in path_list:
        _logger.info("Create segment : %s", _path)
    _logger.info("All time  : %s[s]", _time_all)
    _logger.info("All frame : %s", _frame_all)
