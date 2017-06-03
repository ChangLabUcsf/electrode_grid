# electrode_grid

## Installation
If you want the ability to edit the code locally:
```
git clone https://github.com/ChangLabUcsf/electrode_grid.git
pip install -e electrode_grid
```
If you just want to use it:
```
pip install git+https://github.com/ChangLabUcsf/electrode_grid.git
```

## Usage
```python
from electrode_grid import show_erps, get_channel_order
from scipy.io import loadmat

anatomy_path = os.path.join(subj_dir, 'elecs', 'TDT_elecs_all.mat')
anatomy = np.hstack(loadmat(anatomy_path)['anatomy'][:, -1])
channel_order = get_channel_order('EC61')

Ds = [loadmat('path/to/data.mat')['x'] for x in ('list', 'of', 'matlab', 'variables')]
labels = ('list', 'of', 'condition', 'labels')

show_erps(Ds, labels, anatomy=anatomy, yscale=(-.5, 1.7), channel_order=channel_order)
```
