# Prompt and answer templates of the training mixture.

# Question templates for referring segmentation
SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",
    "What is {class_name} in this image? Please respond with segmentation mask?",
    "What is {class_name} in this image? Please output segmentation mask?",
    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",
]


# Question templates for grounded conversation generation (GCG)
GCG_QUESTIONS = [
    'Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Can you provide a thorough description of this image? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Please describe in detail the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Could you give a comprehensive explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Could you give me an elaborate explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
    'Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
]

# gRefCOCO (GRES) prompt template, used identically in training and eval.
GRES_QUESTION = ("Please segment {class_name} in this image, "
                         "or respond 'No target' if it is not present.")

# Answer templates for referring segmentation, in the '<p> phrase </p> [SEG]' format shared
# with GCG and PanoCaps. '{phrase}' is filled by str.replace.
ANSWER_LIST = [
    "It is <p> {phrase} </p> [SEG].",
    "Sure, <p> {phrase} </p> [SEG].",
    "Sure, it is <p> {phrase} </p> [SEG].",
    "Sure, the segmentation result is <p> {phrase} </p> [SEG].",
    "<p> {phrase} </p> [SEG].",
]


# Question templates for panoptic grounded captioning (PanoCaps, COCONut-PanCap); kept
# lexically distinct from the GCG templates. Template [2] is the PanoCaps eval prompt.
PANOCAPS_QUESTIONS = [
    'Caption this image exhaustively: mention every object and every background region, and insert its segmentation mask right after each mention.',
    'Write a complete scene caption that leaves nothing out: every foreground object and background area must appear, each followed by its segmentation mask.',
    'Write a caption covering all objects and background regions in the image, inserting the matching segmentation mask after each phrase.',
    'Produce a panoptic caption of the scene: name all things and background regions, placing the segmentation mask immediately after each named element.',
    'Give a full-scene caption in which every visible object and background area is mentioned, attaching the segmentation mask after each one.',
    'Compose an exhaustive caption naming every region in the image, foreground and background alike, with the correct segmentation mask after each element you mention.',
]
