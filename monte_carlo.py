import random

# area of a circle definition = pi * r^2 
# area of a the square enclosing it = 4 * r^2 
# area of circle / area of square = pi / 4
# dots inside circle / dots inside square = pi / 4
# (dots inside circle / dots inside square) * 4 = pi 

def estimate_pi(total_points: int) -> float:
    """Estimates PI via Monte Carlo given the total number of points to use."""

    inside = 0
    for _ in range(total_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x ** 2 + y ** 2 < 1:
            inside += 1
    pi = 4 * (inside / total_points)
    return pi

def main(): 
    print(estimate_pi(10000))
    print(estimate_pi(100000))
    print(estimate_pi(1000000))
    print(estimate_pi(10000000))

if __name__ == "__main__":
    main()