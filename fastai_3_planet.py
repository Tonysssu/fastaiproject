from fastai.vision import *
# ! mkdir -p ~/.kaggle/
# ! mv kaggle.json ~/.kaggle/

path = Config.data_path()/'planet'
path.mkdir(parents=True, exist_ok=True)

# ! kaggle competitions download -c planet-understanding-the-amazon-from-space -f train-jpg.tar.7z -p {path}
# ! kaggle competitions download -c planet-understanding-the-amazon-from-space -f train_v2.csv -p {path}
# ! unzip -q -n {path}/train_v2.csv.zip -d {path}
# ! 7za -bd -y -so x {path}/train-jpg.tar.7z | tar xf - -C {path.as_posix()}

df = pd.read_csv(path/'train_v2.csv')
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
