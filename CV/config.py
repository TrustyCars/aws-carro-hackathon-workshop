# Configs for CRAFT
CRAFT_MODEL_PATH = "CRAFT_pytorch/ckpts/craft_mlt_25k.pth"
text_threshold = 0.8
link_threshold = 0.4
low_text = 0.4
poly=False

# Configs for Recognizer
THAI_RECOGNIZER_PATH = "deep_text_recognition_benchmark/saved_models/Thai/best_norm_ED.pth"
ENG_RECOGNIZER_PATH = "deep_text_recognition_benchmark/saved_models/English/best_accuracy.pth"
ENG_PUBLIC_RECOGNIZER_PATH = "deep_text_recognition_benchmark/saved_models/Public/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"

USING_GPU=False
