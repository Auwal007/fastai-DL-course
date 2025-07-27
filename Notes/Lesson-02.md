# Lecture Notes: Chapter 2 – From Model to Production


## 📘 Overview

Chapter 2 takes you beyond training a model and walks through the **end-to-end process of building a deep learning application**, using a **bear classifier** as a practical example. You'll learn how to collect data, train a model, interpret results, build a user interface, and deploy your app online—transforming a Jupyter notebook into a live web application.

> 🐻 Goal: Build a bear classifier that can distinguish between grizzly, black, and teddy bears.

---

## 🔁 The Full Deep Learning Workflow

This chapter emphasizes that **model training is just one step** in a larger pipeline:

1. **Define the objective**
2. **Collect and clean data**
3. **Train a model**
4. **Interpret and debug the model**
5. **Build a user interface (UI)**
6. **Deploy the application**
7. **Share and reflect**

> 💡 "The six lines of code we saw in Chapter 1 are just one small part of the process."

---

## 1. 🎯 Define the Objective

- **Problem**: Classify images of bears into three categories: *grizzly*, *black*, and *teddy*.
- **Use Case**: Could be used in wildlife monitoring, education, or fun personal apps.
- **Key Insight**: Even if your problem doesn’t look like computer vision, it might be transformed into one (e.g., sounds → spectrograms, text → images).

---

## 2. 📥 Collect and Clean Data

### Step 1: Use Search Engines for Data Collection
- Tools: Google Images, Bing Image Search, or `fastai`’s `search_images_dd` (duckduckgo).
- Search terms: `"grizzly bear"`, `"black bear"`, `"teddy bear"`.
- Automate downloads using `download_images()`.

```python
from fastai.vision.utils import *
download_images('grizzly_bear.jpg', dest='grizzly')
download_images('black_bear.jpg', dest='black')
download_images('teddy_bear.jpg', dest='teddy')
```

### Step 2: Remove Invalid or Corrupted Files
```python
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)  # Delete corrupted files
```

> ⚠️ Always verify image files—some may be HTML pages (e.g., access denied), not actual images.

### Step 3: Organize with DataBlock
Use `DataBlock` to define how data is loaded and labeled:

```python
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128)
)
```

| Component | Purpose |
|--------|--------|
| `blocks=(ImageBlock, CategoryBlock)` | Specifies input (image) and output (category) types |
| `get_items` | How to get the list of files |
| `splitter` | Splits data into training (80%) and validation (20%) sets |
| `get_y` | Labels from parent folder names (`grizzly`, `black`, `teddy`) |
| `item_tfms` | Resize all images to 128x128 pixels |

> ✅ Pro tip: Use `parent_label` when folder names are your class labels.

---

## 3. 🏋️ Train the Model

```python
dls = bears.dataloaders(path)
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
```

- Uses **ResNet-18** (faster than ResNet-34 for quick prototyping).
- Trains for 4 epochs.
- Achieves high accuracy quickly thanks to **transfer learning**.

---

## 4. 🔍 Interpret and Debug the Model

Use `ClassificationInterpretation` to analyze performance:

```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```

### Key Tool: `plot_top_losses()`
Shows predictions with the **highest loss** (i.e., where the model was most confident but wrong):

```python
interp.plot_top_losses(5, nrows=1)
```

Each image shows:
- **Prediction**
- **Actual label**
- **Loss value**
- **Prediction probability (confidence)**

> 🔎 Why it matters:
> - Helps identify mislabeled data.
> - Reveals edge cases (e.g., teddy bears in real environments).
> - Guides further data cleaning or augmentation.

---

## 5. 💬 Build a User Interface

Use **Jupyter widgets** to create an interactive app inside the notebook:

```python
import ipywidgets as widgets
uploader = widgets.FileUpload()
uploader
```

Then:
- Load the uploaded image.
- Use `learn.predict()` to classify it.
- Display the result with probability.

```python
img = PILImage.create(uploader.data[0])
pred,_,probs = learn.predict(img)
print(f"Prediction: {pred}; Probability: {probs[0]:.4f}")
```

> 🧩 This turns your model into an interactive tool—ready for real users.

---

## 6. ☁️ Deploy the Application

Deploy your notebook as a **live web app** using:

### Option 1: **Voilà + Binder** (Free & Easy)

#### Steps:
1. Install Voilà:
   ```bash
   !pip install voila
   !jupyter serverextension enable voila --sys-prefix
   ```

2. Mark cells for display:
   - Add tags: `hide_input` to hide code, `voila` to show output only.

3. Deploy via [mybinder.org](https://mybinder.org):
   - Upload your notebook to GitHub.
   - Go to mybinder.org.
   - Enter your repo URL.
   - Set launch path: `/voila/render/notebook_name.ipynb`

4. Click **Launch** → Binder builds environment and serves your app.

> ⏱️ First build takes ~5 minutes. After that, it’s fast.

> 🌐 Result: A shareable URL anyone can use to interact with your model.

### Other Deployment Options:
- **FastAPI + Uvicorn** (for APIs)
- **Streamlit** (simpler UI framework)
- **Hugging Face Spaces**
- **Google Colab + Gradio**

> 📚 Check the book’s website for full guides on alternative deployment methods.

---

## 7. 📣 Share and Reflect

### ✍️ Start a Blog
- Writing solidifies understanding.
- Share your journey: what surprised you? What challenges did you face?
- Example topics:
  - “How I built a bear classifier in one day”
  - “Why my model thought a dog was a teddy bear”
  - “Opportunities for deep learning in my industry”

> 🗣️ "There is no better test of your understanding than teaching it to someone else."

---

## ⚠️ Ethical Considerations

### Models Reflect Their Data
- If training data is biased, the model will be too.
- Example: Searching “healthy skin” returns mostly light-skinned people → model learns bias.
- **Consequence**: Misdiagnosis risk in medical applications.

> 🔁 Key insight: Garbage in, garbage out. Always audit your dataset.

---

## 🛠️ Tools & Tips

| Tool | Purpose |
|------|--------|
| `verify_images()` | Remove corrupted image files |
| `DataBlock` | Flexible way to define data pipelines |
| `ClassificationInterpretation` | Analyze model errors |
| `plot_top_losses()` | Find mislabeled or ambiguous data |
| `Voilà` | Turn notebooks into dashboards |
| `Binder` | Host apps for free in the cloud |

---

## 🧪 Best Practices

1. **Start Small**
   - Use small image sizes (`Resize(128)`) for fast iteration.
   - Scale up later.

2. **Experiment Constantly**
   - Try different architectures (`resnet18`, `resnet34`, etc.).
   - Adjust learning rates, epochs, data splits.

3. **Fail Fast, Learn Faster**
   - Don’t aim for perfection early.
   - Build → Test → Fix → Repeat.

4. **Use the Stripped Notebooks**
   - Close the book and try to re-create the notebook from scratch.
   - Reinforces learning through active recall.

---

## 🔄 Summary: The Full Cycle

| Stage | Key Actions |
|------|------------|
| **Define** | Pick a fun, manageable problem |
| **Collect** | Scrape images, organize folders, clean data |
| **Train** | Use `vision_learner` and `fine_tune` |
| **Interpret** | Use `interp.plot_top_losses()` |
| **Interface** | Add widgets for interaction |
| **Deploy** | Use Voilà + Binder for free hosting |
| **Share** | Write a blog post, reflect, improve |

> 🚀 "You’ve now completed the full cycle from idea to deployed app!"

---

## ✅ Exercises (Suggested)

1. Build your own classifier (e.g., types of birds, cars, plants).
2. Use `plot_top_losses()` to find and fix mislabeled data.
3. Deploy your model using Binder.
4. Write a blog post about your project.
5. Try replacing Jupyter widgets with **Gradio** for a prettier UI.

---

*End of Chapter 2 Notes*