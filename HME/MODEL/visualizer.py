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

args = get_vi_args()
model_path = args.model_path
wav_path = args.wav_path
seg_path = args.seg_path
output_video = args.output_path
cat_video = args.output_pathC


CENTER = np.array([640, 360, 0])

with open("HME/data/visual/facemesh_landmark_list.npz", "rb") as f:
    _face = pickle.load(f)
centroid = np.mean(_face, axis=0)

FACE = _face - centroid + CENTER


SAMPLE_FREQUENCY = 16000
start = int(re.split(r"[_.]", seg_path)[-3])
stop = int(re.split(r"[_.]", seg_path)[-2])

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


# Clip audio
root, ext = os.path.splitext(wav_path)
cut_path = os.path.join(root + "_cut" + ".wav")
sound = AudioSegment.from_file(wav_path, format="wav")
sound1 = sound[frame_term[0] : frame_term[1]]
sound1.export(cut_path, format="wav")

if args.use_model == "small":
    model = SmallModel(
        lstm_dim=16,
        ac_linear_dim=8,
        lstm_input_dim=32,
        lstm_output_dim=16,
        ac_feature_size=80,
        ac_feature_width=32,
        num_layer=2,
    )
elif args.use_model == "simple":
    model = SimpleModel(
        lstm_dim=16,
        ac_linear_dim=8,
        lstm_input_dim=32,
        lstm_output_dim=16,
        ac_feature_size=80,
        ac_feature_width=32,
        num_layer=2,
        pos_feature_size=512,
    )
else:
    raise ValueError(f"Invalid model name, {args.use_model}")

# torch.save(model.state_dict(), model_path)
net_dic = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(net_dic)

pred_angl = angl.clone().detach().requires_grad_(True)
pred_cent = cent.clone().detach().requires_grad_(True)
h, c = None, None

for seq, (trgt, othr) in enumerate(zip(segment[0][2], segment[0][3])):

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
visualize_result(video, result, FACE, output_video, True)

if not args.skip_wav_cat:
    v_clip = mpedit.VideoFileClip(output_video)
    v_clip = v_clip.set_audio(
        mpedit.AudioFileClip(cut_path, fps=frame_rate, nbytes=nbytes)
    )

    v_clip.write_videofile(cat_video, temp_audiofile="HME/MODEL/out/tmp.wav")
