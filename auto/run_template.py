import matplotlib.pyplot as plt
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage


K_P = 1
K_I = 0.3
K_D = 5
STEPS = 550
car = make_car(desired_v=20.0, dt=0.1)


#WRITE CODE HERE
velocities = []
errors = []
times = []


for step in range(STEPS):
   desired_acceleration, error = calculate_desired_acceleration(car, K_P, K_I, K_D)
   throttle_percentage = acceleration_to_throttle_percentage(desired_acceleration)
   velocities.append(car["v"])
   errors.append(error)
   times.append(car["t"])


   update(car, throttle_percentage)


print(f"Final velocity: {car['v']} m/s")


#time vs velocity plot
plt.figure()
plt.plot(times, velocities)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Car Velocity vs Time")
plt.show()


#error vs time plot
plt.figure()
plt.plot(times, errors)
plt.xlabel("Time (s)")
plt.ylabel("Error (m/s)")
plt.title("Car Error vs Time")
plt.show()



