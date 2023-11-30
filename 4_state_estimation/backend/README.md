# graphslam

Implementation of graph based SLAM.

## Setup

### Create a conda virtual environment

```bash
conda create -n slamenv2 python=2.7

```

### Install the dependencies (`requirements.txt` in progress)

```bash
conda install numpy matplotlib 

```

### Activate the virtual environment)

```bash
conda activate slamenv2

```

## Getting Started

### Create graph from TORO or G2O data files

```python
def main():
    graph = load_graph_file('data/file.g2o'))
    
    # Display graph in window.
    graph.plot()

    # Save graph to PDF.
    graph.plot(save=True)


if __name__ == "__main__":
    main()

```
