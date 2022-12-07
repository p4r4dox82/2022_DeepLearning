from transformers import pipeline

BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
MASK = "<unused0>"
Q_TKN = "<unused1>"
S_TKN = {"formal": "<unused10>", 
         "informal":"<unused11>",
         "android": "<unused12>",
         "azae": "<unused13>",
         "chat": "<unused14>",
         "choding": "<unused15>",
         "emoticon": "<unused16>",
         "enfp": "<unused17>",
         "gentle": "<unused18>", 
         "halbae": "<unused19>",
         "halmae": 	"<unused20>",
         "joongding": "<unused21>",
         "king": "<unused22>",
         "naruto": "<unused23>",
         "seonbi": "<unused24>",
         "sosim": "<unused25>",
         "translator": "<unused26>"
}
A_TKN = "<unused3>"
SENT = "<unused4>"

styles = ["formal",
          "informal",
          "android",
          "choding",
          "emoticon",
          "king",
          "naruto",
          "seonbi"
]

selected_model_path = "./text-transfer_11281204/checkpoint-4400/"
model_name = "gogamza/kobart-base-v2"

nlg_pipeline = pipeline('text2text-generation',model=selected_model_path, tokenizer=model_name)

def generate_text(pipe, text, target_style, num_return_sequences=5, max_length=60):
  text = f"{S_TKN[target_style]} translate: {text}"
  out = pipe(text, num_return_sequences=num_return_sequences, max_length=max_length)
  return [x['generated_text'] for x in out]

src_text = "말투가 바뀌긴 하는데 성능이 그렇게 좋은지는 잘 모르겠다 이거 어쩌지"

# print(generate_text(nlg_pipeline, src_text, "chat", num_return_sequences=1, max_length=1000))
print("input : ", src_text)
for style in styles[2:]:
  print(style, generate_text(nlg_pipeline, src_text, style, num_return_sequences=1, max_length=1000)[0])