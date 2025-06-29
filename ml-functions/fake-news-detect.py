import pandas as pd
import os
from PIL import Image
import requests
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from io import BytesIO

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, '..', 'ml-data', 'news_articles.csv')
output_path = os.path.join(script_dir, '..', 'ml-data', 'news_articles_scored.csv')
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Loading spreadsheet
df = pd.read_csv(input_path)

# Loading models
text_classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Calculating reliability score
def compute_reliability(row):
    title = str(row.get("title", ""))
    description = str(row.get("description", ""))
    image_url = row.get("urlToImage", "")

    full_text = f"{title}\n{description}"

    text_score = None
    caption = ""
    caption_score = None

    try:
        text_res = text_classifier(full_text)[0]
        text_score = text_res['score'] if text_res['label'] == 'REAL' else 1 - text_res['score']
    except:
        text_score = None

    try:
        if pd.isna(image_url) or not image_url.startswith("http"):
            raise ValueError("Invalid image URL")
        img = Image.open(requests.get(image_url, stream=True, timeout=5).raw)
        inputs = caption_processor(img, return_tensors="pt")
        outputs = caption_model.generate(**inputs)
        caption = caption_processor.decode(outputs[0], skip_special_tokens=True)

        caption_res = text_classifier(caption)[0]
        caption_score = caption_res['score'] if caption_res['label'] == 'REAL' else 1 - caption_res['score']
    except:
        caption = ""
        caption_score = None

    if text_score is not None and caption_score is not None:
        final_score = round((text_score + caption_score) * 100, 2)
    elif text_score is not None:
        final_score = round(text_score * 100, 2)
    else:
        final_score = None

    return pd.Series({
        "text_score": text_score,
        "caption_score": caption_score,
        "final_score": final_score
    })

# Applying function to each row
results_df = df.apply(compute_reliability, axis=1)
# Combining with original data
df_combined = pd.concat([df, results_df], axis=1)
# Saving results
df_combined.to_csv(output_path, index=False)

print(f"✅ Finished scoring. Saved to: {output_path}")