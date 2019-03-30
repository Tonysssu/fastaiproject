from fastai.vision import *

##urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
## window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));

## 1.training loss less than validation loss : a. not trainging enough  b. learning rate too low or number of epoch too few

## 2.LR too high Or too many epoches will make validation loss extremely high
# Reason : take a really long time to train, it's get too many looks at images, "may" overfit
# Any well tarined model will have training loss < validation loss
## 3. Error rate goes up indicate overfiting (only thing that shows overfiting)

folder = 'happy'
path = Path('data/emotion')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)

file = 'urls_happy.txt'
classes = ['anger', 'sad', 'happy','surprise']
download_images(path/file, dest, max_pics=200)

for c in classes:
    print(c)
    verify_images(path/c, delete=False, max_size=500)

np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
    ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
#data.classes
#data.show_batch(rows=3, figsize=(7,8))
#data.classes, data.c, len(data.train_ds), len(data.valid_ds)

#Training
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(8)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8, max_lr=slice(3e-3,1e-2))
learn.save('stage-2')

#Interpretation
learn.load('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

#Data cleaning
