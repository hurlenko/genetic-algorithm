# Genetic Algorithm

Simple implementation of Genetic Algorithm written in python3. A genetic algorithm is an algorithm that imitates the process of natural selection.

## Supported operators

### Crossover

1. Single point
2. Multi point
3. Uniform
4. Shuffling
5. Reduced surrogate

### Parent selection

1. Panmixia
2. Inbreeding (Euclidean/Hamming distance)
3. Outbreeding (Euclidean/Hamming distance)
4. Tournament
5. Roulette wheel

### Offspring selection

1. Truncution
2. Elite
3. Exclusion
4. Bolzmann

### Mutators

1. Single point

## Usage

- Install Numpy

```bash
pip install numpy
```

- Tweak constants in `main.py`, fitness function, number of iterations etc.
- Run

```python
python main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details