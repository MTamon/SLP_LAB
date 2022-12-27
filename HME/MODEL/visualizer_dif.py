"""This program is for visualize results"""

import pickle
import wave
import os
import re
import numpy as np
import torch
import moviepy.editor as mpedit
from pydub import AudioSegment

from ohta_dataloader import OhtaDataloader
from src import Video, visualize_result, get_vi_args
from model.ohta.alphaS import AlphaS


CENTER = np.array([640, 360, 0])

with open("HME/data/visual/facemesh_landmark_list.npz", "rb") as f:
    _face = pickle.load(f)
centroid = np.mean(_face, axis=0)

FACE = _face - centroid + CENTER


def visual_process(
    seg_path, wav_path, out_site, tmp_path, use_model, model_path, skip_wav_cat=False
):
    start = int(re.split(r"[_.]", seg_path)[-3])
    stop = int(re.split(r"[_.]", seg_path)[-2])

    fname = os.path.basename(seg_path)
    output_video = out_site + "/" + re.split(r"[.]", fname)[0] + ".mp4"
    cat_video = out_site + "/" + re.split(r"[.]", fname)[0] + "C.mp4"

    segment = []
    o_angl = []
    o_cent = []
    for i in range(29, stop - start, 1):
        _data = OhtaDataloader.get_data(
            datasite=os.path.dirname(seg_path),
            file_pointer=(os.path.basename(seg_path), i),
            acostic_frame_width=69,
            physics_frame_width=10,
            acostic_dim=80,
        )
        if not o_angl and not o_cent:
            o_angl = [rec for rec in _data[0][0][0]]
            o_cent = [rec for rec in _data[0][2][0]]
        o_angl.append(_data[0][0][0][-1] + _data[1][0][0])
        o_cent.append(_data[0][2][0][-1] + _data[1][1][0])
        _data = [
            [
                [_data[0][0][0].unsqueeze(0), _data[0][1][0].unsqueeze(0)],  # angle
                [_data[0][2][0].unsqueeze(0), _data[0][3][0].unsqueeze(0)],  # cent
                [_data[0][4][0].unsqueeze(0), _data[0][5][0].unsqueeze(0)],  # target
                [_data[0][6][0].unsqueeze(0), _data[0][7][0].unsqueeze(0)],  # other
                [_data[0][8][0].unsqueeze(0), _data[0][9][0].unsqueeze(0)],  # t-power
                [_data[0][10][0].unsqueeze(0), _data[0][11][0].unsqueeze(0)],  # o-power
            ],
            [
                _data[1][0][0].unsqueeze(0),
                _data[1][1][0].unsqueeze(0),
            ],
        ]
        segment.append(_data)
    with wave.open(wav_path, "rb") as wav:
        num_samples = wav.getnframes()
        frame_rate = wav.getframerate()
        nbytes = wav.getsampwidth()
        waveform = wav.readframes(num_samples)
        waveform = np.frombuffer(waveform, dtype=np.int16)
    with open(seg_path, "rb") as seg_f:
        seg = pickle.load(seg_f)
        vfps = seg["vfps"]

    o_angl = torch.stack(o_angl)
    o_cent = torch.stack(o_cent)

    TERM = np.array([start + 29, stop]) / vfps
    frame_term = (TERM * 1000).astype(np.int32)

    o_angl = (
        o_angl.view([-1, 3])
        .to(dtype=torch.float32)
        .clone()
        .detach()
        .requires_grad_(True)
    )
    o_cent = (
        o_cent.view([-1, 3])
        .to(dtype=torch.float32)
        .clone()
        .detach()
        .requires_grad_(True)
    )
    original = (o_angl, o_cent)

    # Clip audio
    seg_name = os.path.basename(seg_path)
    cut_path = tmp_path + "/" + seg_name.split(".")[0] + "_cut.wav"
    sound = AudioSegment.from_file(wav_path, format="wav")
    sound1 = sound[frame_term[0] : frame_term[1]]
    sound1.export(cut_path, format="wav")

    if use_model == "alphaS":
        model = AlphaS(
            acostic_dim=80,
            num_layers=2,
            kernel_size=5,
            out_kernel_size=3,
            num_channels=64,
            cnet_out_dim=160,
            encoder_dim=256,
            physic_frame_width=10,
            acostic_frame_width=69,
        )
    else:
        raise ValueError(f"Invalid model name, {use_model}")

    net_dic = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(net_dic)

    pred_angl = segment[0][0][0][0][0].clone().detach().requires_grad_(True)
    delt_angl = segment[0][0][0][1][0].clone().detach().requires_grad_(True)
    pred_cent = segment[0][0][1][0][0].clone().detach().requires_grad_(True)
    delt_cent = segment[0][0][1][1][0].clone().detach().requires_grad_(True)

    for dt in segment:

        dt[0][0][0] = pred_angl[-10:].unsqueeze(0)
        dt[0][0][1] = delt_angl[-10:].unsqueeze(0)
        dt[0][1][0] = pred_cent[-10:].unsqueeze(0)
        dt[0][1][1] = delt_cent[-10:].unsqueeze(0)

        with torch.no_grad():
            (_d_angl, _d_cent) = model(dt[0])

        next_angl = pred_angl[-1:] + _d_angl
        next_cent = pred_cent[-1:] + _d_cent

        pred_angl = torch.cat((pred_angl, next_angl), axis=0)
        delt_angl = torch.cat((delt_angl, _d_angl), axis=0)
        pred_cent = torch.cat((pred_cent, next_cent), axis=0)
        delt_cent = torch.cat((delt_cent, _d_cent), axis=0)

    result = (pred_angl, pred_cent)
    video = Video("HME/data/visual/K002_018_GP02.mp4")
    visualize_result(video, original, result, FACE, output_video, True)

    if not skip_wav_cat:
        v_clip = mpedit.VideoFileClip(output_video)
        v_clip = v_clip.set_audio(
            mpedit.AudioFileClip(cut_path, fps=frame_rate, nbytes=nbytes)
        )

        v_clip.write_videofile(cat_video, temp_audiofile="HME/MODEL/out/tmp.wav")


if __name__ == "__main__":
    args = get_vi_args()
    _inputs = args.inputs
    _use_model = args.use_model
    _model_path = args.model_path
    _out_site = args.outsite
    _skip_wav_cat = args.skip_wav_cat
    _tmp_path = args.tmp_path

    dir_list = os.listdir(_inputs)
    file_list = []
    for direc in dir_list:
        fset = {"seg": None, "wav": None}
        direc_path = _inputs + "/" + direc
        direc_mem = os.listdir(direc_path)
        if len(direc_mem) != 2:
            raise ValueError(f"{direc_path} have {len(direc_mem)} members.")

        for file in direc_mem:
            if file.split(".")[-1] == "seg":
                fset["seg"] = direc_path + "/" + file
            elif file.split(".")[-1] == "wav":
                fset["wav"] = direc_path + "/" + file
            else:
                raise ValueError("Detect not .wav .mp4 file.")

        if fset["seg"] is None or fset["wav"] is None:
            raise ValueError(f"{direc_path} lack member.")
        file_list.append(fset)

    for file in file_list:
        visual_process(
            file["seg"],
            file["wav"],
            _out_site,
            _tmp_path,
            _use_model,
            _model_path,
            _skip_wav_cat,
        )
