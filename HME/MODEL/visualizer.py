"""This program is for visualize results"""

import pickle
import wave
import os
import re
import numpy as np
import torch
import moviepy.editor as mpedit
from pydub import AudioSegment

from src import get_data, Video, visualize_result, get_vi_args
from model import SmallModel
from model import SimpleModel


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

    segment = get_data(seg_path, ac_feature_size=80, ac_feature_width=32)
    with wave.open(wav_path, "rb") as wav:
        num_samples = wav.getnframes()
        frame_rate = wav.getframerate()
        nbytes = wav.getsampwidth()
        waveform = wav.readframes(num_samples)
        waveform = np.frombuffer(waveform, dtype=np.int16)

    angl = segment[0][0][0].view([1, 1, 3]).to(dtype=torch.float32)
    cent = segment[0][1][0].view([1, 1, 3]).to(dtype=torch.float32)
    TERM = np.array([start, stop]) / segment[2]
    frame_term = (TERM * 1000).astype(np.int32)

    o_angl = (
        segment[1][0]
        .view([-1, 3])
        .to(dtype=torch.float32)
        .clone()
        .detach()
        .requires_grad_(True)
    )
    o_cent = (
        segment[1][1]
        .view([-1, 3])
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

    if use_model == "small":
        model = SmallModel(
            lstm_dim=16,
            ac_linear_dim=8,
            lstm_input_dim=32,
            lstm_output_dim=16,
            ac_feature_size=80,
            ac_feature_width=32,
            num_layer=2,
        )
    elif use_model == "simple":
        model = SimpleModel(
            lstm_dim=1024,
            ac_linear_dim=512,
            lstm_input_dim=2048,
            lstm_output_dim=2048,
            ac_feature_size=80,
            ac_feature_width=32,
            num_layer=2,
            pos_feature_size=10,
        )
    else:
        raise ValueError(f"Invalid model name, {use_model}")

    # torch.save(model.state_dict(), model_path)
    net_dic = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(net_dic)

    pred_angl = angl.clone().detach().requires_grad_(True)
    pred_cent = cent.clone().detach().requires_grad_(True)
    h, c = None, None

    for (trgt, othr) in zip(segment[0][2], segment[0][3]):

        trgt = trgt.view([1, 1, -1]).to(dtype=torch.float32)
        othr = othr.view([1, 1, -1]).to(dtype=torch.float32)

        model_input = [angl, cent, trgt, othr]

        with torch.no_grad():
            (_p_angl, _p_cent), (h, c) = model(model_input, h, c)

        pred_angl = torch.cat((pred_angl, _p_angl), axis=1)
        pred_cent = torch.cat((pred_cent, _p_cent), axis=1)

        angl, cent = _p_angl, _p_cent

    result = (pred_angl[0], pred_cent[0])
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
