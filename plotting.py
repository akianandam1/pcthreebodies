#from TorchNumericalSolver import torchIntegrate
from RevisedNumericalSolver import get_full_state
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import torch
import time
from datetime import datetime
from nhd import data

mpl.rcParams['animation.ffmpeg_path'] = r'D:\MAIN\Python37\Lib\site-packages\ffmpeg-5.0-essentials_build\bin\ffmpeg.exe'


# input_vec = torch.tensor([2.3050e-01, -4.0595e-02, 8.5479e-44])
# full_vec = torch.cat(
#     [torch.tensor([1, 0, 0, 0, 0, 0, 0, 1, 0]), input_vec, torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])])
#full_vec = torch.tensor([-1, 0, 0, 1, 0, 0, 0, 0, 0, 0.347111, 0.532728, 0, 0.347111, 0.532728, 0, -2*0.347111, -2*0.532728, 0, 1, 1, 1])
# m_1 = .3916
# m_2 = .8341
# m_3 = 1
# x_1 = -1.21503
# v_1 = -1.00328
# v_2 = -.53749
input_set = data[0]
m_1 = float(input_set[0])
m_2 = float(input_set[1])
m_3 = float(input_set[2])
x_1 = float(input_set[3])
v_1 = float(input_set[4])
v_2 = float(input_set[5])
T = float(input_set[6])
full_vec = torch.tensor([x_1, 0, 0, 1, 0, 0, 0, 0, 0, 0, v_1, 0, 0, v_2, 0, 0, -(m_1 * v_1 + m_2 * v_2) / m_3, 0, m_1, m_2, m_3],
                   requires_grad = True)

# m_1 = 1
# m_2 = 1
# m_3 = 0.5
# v_1 = 0.2869236336
# v_2 = 0.0791847624
# #full_vec = torch.tensor([x_1, 0, 0, 1, 0, 0, 0, 0, 0, 0, v_1, 0, 0, v_2, 0, 0, -(m_1*v_1+m_2*v_2)/m_3, 0, m_1, m_2, m_3], requires_grad=True)

#full_vec = torch.tensor([-1,0,0,1,0,0,0,0,0,v_1, v_2, 0, v_1, v_2, 0, -2*v_1/m_3, -2*v_2/m_3, 0, m_1, m_2, m_3], requires_grad = True)


# full_vec = torch.tensor([ 1.5305e+00,  7.2942e-01, -7.9352e-02, -1.7411e-01,  4.6347e-01,
#         -7.3587e-02, -6.5644e-01, -3.9288e-01, -4.7061e-02,  6.4761e-01,
#          5.2369e-01, -6.4135e-03,  1.9012e-01,  2.8642e-01, -7.6145e-03,
#         -7.9527e-01, -7.8276e-01,  1.3103e-02,  3.5702e+01,  3.5733e+01,
#          3.5693e+01])
# start = time.time()
#
# euler_sol = torchIntegrate(full_vec, .001, 100).numpy()
# end = time.time()
# print(f"Euler: {end-start}")
start = time.time()
print("Getting...")
runge_sol = get_full_state(full_vec, .00025, int(T+2)).detach().numpy()
end = time.time()
print(f"Runge Kutta: {end-start}")


first = time.time()
# euler_r1_sol = euler_sol[:, 0:3]
# euler_r2_sol = euler_sol[:, 3:6]
# euler_r3_sol = euler_sol[:, 6:9]
runge_r1_sol = runge_sol[:, 0:3]
runge_r2_sol = runge_sol[:, 3:6]
runge_r3_sol = runge_sol[:, 6:9]


# INITIAL POSITIONS: (-1,0), (1,0), (0,0)
#
# INITIAL VELOCITIES: (p1,p2), (p1,p2), (-2p1,-2p2)
#
# p1: 0.347111
#
# p2: 0.532728



# Create figure
fig = plt.figure(figsize=(20, 20))  # Create 3D axes
ax = fig.add_subplot(111, projection="3d")  # Plot the orbits
ax.set_zlim(-3, 3)
ax.set_title("Visualization of orbits of stars in a three body system\n", fontsize=28)

# euler_particle1, = plt.plot([], [], [], color='r')
# euler_particle2, = plt.plot([], [], [], color='r')
# euler_particle3, = plt.plot([], [], [], color='r')
runge_particle1, = plt.plot([], [], [], color='r')
runge_particle2, = plt.plot([], [], [], color='g')
runge_particle3, = plt.plot([], [], [], color='b')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# p1, = plt.plot([], [], marker='o', color='r', label="Real first star")
# p2, = plt.plot([], [], marker='o', color='g', label="Real second star")
# p3, = plt.plot([], [], marker='o', color='b', label="Real third star")


# model_particle1, = plt.plot([], [], [], color='r')
# model_particle2, = plt.plot([], [], [], color='g')
# model_particle3, = plt.plot([], [], [], color='b')
#
# model_p1, = plt.plot([], [], marker='s', color='r', label="Model's first star")
# model_p2, = plt.plot([], [], marker='s', color='g', label="Model's second star")
# model_p3, = plt.plot([], [], marker='s', color='b', label="Model's third star")

#ax.legend(loc="upper left", fontsize=28)
def update(i):
    i*=10
    # euler_particle1.set_data(euler_r1_sol[:i, 0], euler_r1_sol[:i, 1])
    # euler_particle1.set_3d_properties(euler_r1_sol[:i, 2])
    # euler_particle2.set_data(euler_r2_sol[:i, 0], euler_r2_sol[:i, 1])
    # euler_particle2.set_3d_properties(euler_r2_sol[:i, 2])
    # euler_particle3.set_data(euler_r3_sol[:i, 0], euler_r3_sol[:i, 1])
    # euler_particle3.set_3d_properties(euler_r3_sol[:i, 2])

    runge_particle1.set_data(runge_r1_sol[:i, 0], runge_r1_sol[:i, 1])
    runge_particle1.set_3d_properties(runge_r1_sol[:i, 2])
    runge_particle2.set_data(runge_r2_sol[:i, 0], runge_r2_sol[:i, 1])
    runge_particle2.set_3d_properties(runge_r2_sol[:i, 2])
    runge_particle3.set_data(runge_r3_sol[:i, 0], runge_r3_sol[:i, 1])
    runge_particle3.set_3d_properties(runge_r3_sol[:i, 2])

    # p1.set_data(r1_sol[i:i + 1, 0], r1_sol[i:i + 1, 1])
    # p1.set_3d_properties(r1_sol[i:i + 1, 2])
    # p2.set_data(r2_sol[i:i + 1, 0], r2_sol[i:i + 1, 1])
    # p2.set_3d_properties(r2_sol[i:i + 1, 2])
    # p3.set_data(r3_sol[i:i + 1, 0], r3_sol[i:i + 1, 1])
    # p3.set_3d_properties(r3_sol[i:i + 1, 2])
    print(i)

    # model_particle1.set_data(model_r1_sol[:i, 0], model_r1_sol[:i, 1])
    # model_particle1.set_3d_properties(model_r1_sol[:i, 2])
    # model_particle2.set_data(model_r2_sol[:i, 0], model_r2_sol[:i, 1])
    # model_particle2.set_3d_properties(model_r2_sol[:i, 2])
    # model_particle3.set_data(model_r3_sol[:i, 0], model_r3_sol[:i, 1])
    # model_particle3.set_3d_properties(model_r3_sol[:i, 2])
    #
    # model_p1.set_data(model_r1_sol[i:i + 1, 0], model_r1_sol[i:i + 1, 1])
    # model_p1.set_3d_properties(model_r1_sol[i:i + 1, 2])
    # model_p2.set_data(model_r2_sol[i:i + 1, 0], model_r2_sol[i:i + 1, 1])
    # model_p2.set_3d_properties(model_r2_sol[i:i + 1, 2])
    # model_p3.set_data(model_r3_sol[i:i + 1, 0], model_r3_sol[i:i + 1, 1])
    # model_p3.set_3d_properties(model_r3_sol[i:i + 1, 2])

    return runge_particle1, runge_particle2, runge_particle3, #euler_particle1, euler_particle2, euler_particle3,model_particle1, model_particle2, model_particle3, model_p1, model_p2, model_p3


video_folder="AfterJuly18"
now = str(datetime.now())
video_title = now[:now.index(".")]
video_title = video_title[:video_title.index(" ")] + "-" + video_title[video_title.index(" ") + 1:]
video_title = video_title[:video_title.index(":")] + "-" + video_title[video_title.index(":") + 1:]
video_title = video_title[:video_title.index(":")] + "-" + video_title[video_title.index(":") + 1:]

writer = animation.FFMpegWriter(fps=100)
ani = animation.FuncAnimation(fig, update, frames=int(T+2)*400, interval=25, blit=True)
#ani.save(r"D:\Main\PycharmProjects\PeriodicThreeBodies\Videos\July12\a1.mp4", writer=writer)
ani.save(f"D:\\Main\\PycharmProjects\\PeriodicThreeBodies\\Videos\\{video_folder}\\a{video_title}.mp4",
              writer = writer)

second = time.time()
plot_time = second-first

print(f"Plotting: {plot_time}")