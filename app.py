import gradio as gr
from fastai.vision.all import *
learn = load_learner('hotdog.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = 'Hotdog or not'
description = 'Jin Yang'
article = "<p style = 'text-align: center'><a href='https://www.youtube.com/watch?v=ACmydtFDTGs' target='_blank'>Inspiration</a></p>"
examples = ['hotdog and nothotdog.png']
interpretation='default'

import gradio as gr
gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=2),title=title,description=description,article=article,examples=examples,interpretation=interpretation).launch(share=True)
