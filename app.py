import gradio as gr
from fairseq import checkpoint_utils, tasks
import torch
import torchaudio
from fairseq.sequence_generator import SequenceGenerator

# Load model 
model_path = "/mnt/data/xixu/runs/WACO/waco_mt_10h_ft_1h/checkpoint_best.pt" 
models, cfg = checkpoint_utils.load_model_ensemble([model_path])
model = models[0]
task = tasks.setup_task(cfg.task)
src_dict = task.source_dictionary
tgt_dict = task.target_dictionary
lang_prefix_tok = getattr(cfg.task, 'lang_prefix_tok', None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Gradio interface
def translate(audio, mic_audio=None):
    if mic_audio is not None:
        input_audio = mic_audio
        input_type = "microphone"
    elif audio is not None:
        input_audio = audio
        input_type = "audio"
    else:
        return "please provide audio"

    waveform, sr = torchaudio.load(input_audio)
    feat_lengths = torch.tensor([waveform.size(1)])
    prev_output_tokens = torch.LongTensor([[tgt_dict.bos()]])
    # debug to check whether the input is correct
    print("wave shape: ", waveform.shape)
    print("input path: ", input_audio)
    print("input type:", input_type)

    sample = {
        "net_input": {
        "src_tokens": waveform.to(device),
        "src_lengths": feat_lengths.to(device),
        "prev_output_tokens": prev_output_tokens.to(device)  
        }
    }

    bsz, _ = sample["net_input"]["src_tokens"].size()[:2]

    if lang_prefix_tok is not None:
        prefix_index = tgt_dict.index(lang_prefix_tok)
        assert prefix_index != tgt_dict.unk_index, "The language prefix token is not in the dictionary."
        prefix_tokens = torch.LongTensor([prefix_index]).to(device).unsqueeze(1).expand(bsz, -1)
    else:
        prefix_tokens = None

    generator = SequenceGenerator(models=models, tgt_dict=tgt_dict, beam_size=10,len_penalty=0.3) 
    translations = generator.generate(model, sample, prefix_tokens=prefix_tokens)

    translation = tgt_dict.string(translations[0][0]["tokens"])

    if lang_prefix_tok is not None:
        translation = translation.replace(lang_prefix_tok + ' ', "")
    # Convert Fairseq's space token `▁`
    translation = translation.replace('▁', ' ').strip()

    return translation

gr.Interface(
    fn=translate,
    inputs=[    
        gr.Audio(type="filepath", label="Upload Audio"),
        gr.Microphone(label="Record Speech", sources="microphone", type="filepath")
        ],
    outputs="text",
    examples=[
        ["/mnt/data/xixu/datasets/must-c-v1.0/mt-en/data/st/wav/Dia_Spont02_13M_14M_spk_1_15.wav", None],
        ["/mnt/data/xixu/datasets/must-c-v1.0/mt-en/data/st/wav/Dia_Spont02_13M_14M_spk_0_23.wav", None],
        ["/mnt/data/xixu/datasets/must-c-v1.0/mt-en/data/st/wav/Mono_Task_MMap02_13M_spk_0_134.wav", None],
        ["/mnt/data/xixu/datasets/must-c-v1.0/mt-en/data/st/wav/Mono_Task_MMap02_13M_spk_0_117.wav", None],
        ["/mnt/data/xixu/datasets/must-c-v1.0/mt-en/data/st/wav/Mono_Discuss_Topic03_07M_spk_0_79.wav", None]
        ],
    cache_examples=True,
    title="WACO: Maltese to English Translation"
).launch(share=True)