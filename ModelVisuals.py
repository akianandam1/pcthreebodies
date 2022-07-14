from RevisedNumericalSolver import get_full_state as runge_state
from TorchNumericalSolver import get_full_state
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import torch
from random import random

mpl.rcParams['animation.ffmpeg_path'] = r'D:\MAIN\Python37\Lib\site-packages\ffmpeg-5.0-essentials_build\bin\ffmpeg.exe'


# Function to determine whether gpu is available or not
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


# Automatically puts data onto the default device
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



def forward(input_vec, time_step, time_length):
    return get_full_state(input_vec, time_step, time_length)
def runge_forward(input_vec, time_step, time_length):
    return runge_state(input_vec, time_step, time_length)
# Epoch ~6400 at .001 time length 15

# Compares two states and returns a numerical value rating how far apart the two states in the three
# body problem are to each other. Takes in two tensor states and returns tensor of value of distance rating
# The higher the score, the less similar they are
def nearest_position(particle, state1, state2):
    mse = torch.nn.L1Loss()
    if particle == 1:
        return mse(state1[:3], state2[:3]) + mse(state1[9:12], state2[9:12])
    elif particle == 2:
        return mse(state1[3:6], state2[3:6]) + mse(state1[12:15], state2[12:15])
    elif particle == 3:
        return mse(state1[6:9], state2[6:9]) + mse(state1[15:18], state2[15:18])
    else:
        print("bad input")


# Finds the most similar state to the initial position in a data set
def nearest_position_state(particle, state, data_set, min, max, time_step):
    i = min
    max_val = torch.tensor([100000000]).to(device)
    index = -1
    while i < max:
        if nearest_position(particle, state, data_set[i]).item() < max_val.item():
            index = i
            max_val = nearest_position(particle, state, data_set[i])

        i += 1
    print(f"Time: {index*time_step}")
    return index


figure, ax = plt.subplots(2, 1)
top, bottom = ax
top.set_xlim(-3,3)
top.set_ylim(-3,3)
# fig = plt.figure(figsize=(20, 20))
# ax = plt.axes(xlim=(-3, 3), ylim=(-3, 3))
# ax.set_title("Visualization of orbits of stars in a three body system\n", fontsize=28)
particle1, = top.plot([], [], color='r', label="Real First Star")
particle2, = top.plot([], [], color='g', label="Real Second Star")
particle3, = top.plot([], [], color='b', label="Real Third Star")
first_text = bottom.text(0.7, 0.85, "", fontsize = "xx-small", transform=ax[1].transAxes)
second_text = bottom.text(0.7, 0.78, "", fontsize = "xx-small", transform = ax[1].transAxes)
third_text = bottom.text(0.7, 0.71, "", fontsize = "xx-small", transform = ax[1].transAxes)
step_text = bottom.text(0.7, 0.64, "", fontsize = "xx-small", transform = ax[1].transAxes)

# ax.legend(loc="upper left", fontsize=28)
top.legend(loc="upper left", fontsize=6)

# Starting initial vector
#vec = torch.tensor([-1, 0, 0, 1, 0, 0, 0, 0, 0, 0.347111, 0.532728, 0, 0.347111, 0.532728, 0, -2*0.347111, -2*0.532728, 0, 35.7071, 35.7071, 35.7071], requires_grad = True).to(device)
# vec = torch.tensor([-1.0054e+00,  1.5448e-04,  0.0000e+00,  9.9720e-01, -6.9657e-03,
#          0.0000e+00,  8.2357e-03,  6.8113e-03,  0.0000e+00,  3.4835e-01,
#          5.2141e-01,  0.0000e+00,  3.3350e-01,  5.3554e-01,  0.0000e+00,
#         -6.8331e-01, -1.0520e+00,  0.0000e+00,  3.5707e+01,  3.5711e+01,
#          3.5704e+01], requires_grad=True).to(device)

# vec = torch.tensor([-1.0064e+00,  2.3667e-03,  0.0000e+00,  9.9728e-01, -7.4266e-03,
#          0.0000e+00,  9.1086e-03,  5.0600e-03,  0.0000e+00,  3.4699e-01,
#          5.2103e-01,  0.0000e+00,  3.3373e-01,  5.3466e-01,  0.0000e+00,
#         -6.8067e-01, -1.0512e+00,  0.0000e+00,  3.5706e+01,  3.5713e+01,
#          3.5701e+01], requires_grad=True).to(device)

# vec = torch.tensor([ 1.5305e+00,  7.2942e-01, -7.9352e-02, -1.7411e-01,  4.6347e-01,
#         -7.3587e-02, -6.5644e-01, -3.9288e-01, -4.7061e-02,  6.4761e-01,
#          5.2369e-01, -6.4135e-03,  1.9012e-01,  2.8642e-01, -7.6145e-03,
#         -7.9527e-01, -7.8276e-01,  1.3103e-02,  3.5702e+01,  3.5733e+01,
#          3.5693e+01], requires_grad=True).to(device)


def init():
    particle1, = top.plot([], [], color='r', label="Real First Star")
    particle2, = top.plot([], [], color='g', label="Real Second Star")
    particle3, = top.plot([], [], color='b', label="Real Third Star")
    first_text = bottom.text(0.7, 0.85, "", fontsize = "xx-small", transform=ax[1].transAxes)
    second_text = bottom.text(0.7, 0.78, "", fontsize = "xx-small", transform = ax[1].transAxes)
    third_text = bottom.text(0.7, 0.71, "", fontsize = "xx-small", transform = ax[1].transAxes)
    step_text = bottom.text(0.7, 0.64, "", fontsize = "xx-small", transform = ax[1].transAxes)
    return particle1, particle2, particle3, first_text, second_text, third_text


loss_values = []


def update(i, lr, time_step, num_stages, max_period, vec, m_1, m_2, m_3):
    input_vec = torch.cat((vec, torch.tensor([m_1,m_2,m_3])))

    # max_period = 15
    # if i==0:
    #     data_set = forward(input_vec, .01, max_period)
    #     step = .01
    #     step_text_val = f"dt: .{step}"
    #     first_period = nearest_position_state(1, data_set[0], data_set, 300, len(data_set), step)*.01
    #     second_period = nearest_position_state(2, data_set[0], data_set, 300, len(data_set), step)*.01
    #     third_period = nearest_position_state(3, data_set[0], data_set, 300, len(data_set), step)*.01
    #     max_period = int(max(first_period, second_period, third_period)) + 1
    #     print(f"Max Period: {max_period}")
    #
    # elif i < num_stages:
    #     data_set = forward(input_vec,.01, max_period)
    #     step = .01
    #     step_text_val = f"dt: .{step}"
    # elif i < num_stages*2:
    #     data_set = forward(input_vec, .007, max_period)
    #     step = .007
    #     step_text_val = f"dt: .{step}"
    # elif i < num_stages*3:
    #     data_set = forward(input_vec, .004, max_period)
    #     step = .004
    #     step_text_val = f"dt: .{step}"
    # elif i < num_stages * 4:
    #     data_set = runge_forward(input_vec, .004, max_period)
    #     step = .004
    #     step_text_val = f"dt: .{step} (Runge Kutta)"
    # else:
    #     data_set = runge_forward(input_vec, .002, max_period)
    #     step = .002
    #     step_text_val = f"dt: .{step} (Runge Kutta)"


    if False:
        data_set = runge_forward(input_vec, .004, max_period)
        step = .004
        step_text_val = f"dt: .{step}"
        first_period = nearest_position_state(1, data_set[0], data_set, 300, len(data_set), step) * step
        second_period = nearest_position_state(2, data_set[0], data_set, 300, len(data_set), step) * step
        third_period = nearest_position_state(3, data_set[0], data_set, 300, len(data_set), step) * step
        max_period = int(max(first_period, second_period, third_period)) + 1
        print(f"Max Period: {max_period}")
    else:
        data_set = runge_forward(input_vec, time_step, max_period)
        step = time_step
        step_text_val = f"dt: .{step} (Runge Kutta)"
        print(f"Max Period: {max_period}")

    #optimizer = torch.optim.Adam([input_vec], lr = lr)
    optimizer = torch.optim.Adam([vec], lr = lr)
    optimizer.zero_grad()

    #data_set = forward(input_vec)
    first_index = nearest_position_state(1, data_set[0], data_set, 300, len(data_set), step)
    first_particle_state = data_set[first_index]
    second_index = nearest_position_state(2, data_set[0], data_set, 300, len(data_set), step)
    second_particle_state = data_set[second_index]
    third_index = nearest_position_state(3, data_set[0], data_set, 300, len(data_set), step)
    third_particle_state = data_set[third_index]
    loss = nearest_position(1, data_set[0], first_particle_state) + nearest_position(2, data_set[0],
                                                                                     second_particle_state) + nearest_position(
        3, data_set[0], third_particle_state)

    print(" ")
    loss_values.append(loss.item())

    print(loss)

    loss.backward()
    print(vec.grad)
    # Updates input vector
    optimizer.step()
    print(input_vec)

    data_set = data_set.cpu().detach().numpy()

    particle1.set_data(data_set[:, 0], data_set[:, 1])
    particle2.set_data(data_set[:, 3], data_set[:, 4])
    particle3.set_data(data_set[:, 6], data_set[:, 7])

    first_text.set_text(f"First Particle Time: {round(first_index*step, 2)}")
    second_text.set_text(f"Second Particle Time: {round(second_index*step, 2)}")
    third_text.set_text(f"Third Particle Time: {round(third_index*step, 2)}")
    step_text.set_text(step_text_val)

    bottom.plot([x for x in range(len(loss_values))], loss_values, color="red")

    print(f"Epoch:{i}")
    print(" ")
    return particle1, particle2, particle3, first_text, second_text, third_text



num_epochs = 30
learning_rate = .0001
len_stages = num_epochs / 5

# vec = torch.tensor([-1, 0, 0, 1, 0, 0, 0, 0, 0, 0.347111, 0.532728, 0, 0.347111, 0.532728, 0, -2*0.347111, -2*0.532728, 0], requires_grad = True)

m_1 = 10.3501 #1.0124 #.3916
m_2 = 10.4522 #0.9968 #.8341
m_3 = 1
x_1 = -1.69797 #-1.32962 #-1.21503
v_1 = -1.22777#-0.88963 #-1.00328
v_2 = 0.86183 #-0.28501 #-.53749

#vec = torch.tensor([x_1, 0, 0, 1, 0, 0, 0, 0, 0, 0, v_1, 0, 0, v_2, 0, 0, -(m_1*v_1+m_2*v_2)/m_3, 0], requires_grad=True)
vec = torch.tensor([-1.6545e+00,  3.5054e-02,  0.0000e+00,  1.0266e+00,  9.4862e-03,
         0.0000e+00,  4.7751e-02,  6.1410e-03,  0.0000e+00,  1.3916e-02,
        -1.2146e+00,  0.0000e+00, -1.4885e-02,  8.5641e-01,  0.0000e+00,
         3.5063e-02,  3.7305e+00,  0.0000e+00], requires_grad=True)
# perturbance = torch.tensor([.05, .05, 0, .05, .05, 0, .05, .05, 0, .01, .01, 0, .01, .01, 0, .01, .01, 0])
perturbance = torch.tensor([random(), random(), 0, random(), random(), 0, random(), random(), 0, random(), random(), 0, random(), random(), 0, random(), random(), 0])
# perturbance *= .05
# print(perturbance)
# with torch.no_grad():
#     vec += perturbance


# vec = torch.tensor([-0.9493,  0.0502,  0.0000,  1.0504,  0.0511,  0.0000,  0.0489,  0.0488,
#          0.0000,  0.3462,  0.5332,  0.0000,  0.3488,  0.5319,  0.0000, -0.6939,
#         -1.0644,  0.0000], requires_grad = True)



writer = animation.FFMpegWriter(fps=int(num_epochs/10))
ani = animation.FuncAnimation(figure, update, frames=num_epochs, fargs=(learning_rate, .002, len_stages, 10, vec, m_1, m_2, m_3))
ani.save(r"D:\Main\PycharmProjects\PeriodicThreeBodies\Videos\July12\a14.mp4", writer=writer)



with open("mainoutput.txt", "a") as file:
    file.write("\n")
    file.write(f"{num_epochs}, {learning_rate}: \n")
    file.write(str(vec))

print(vec)

# reset_input = input("Press enter to reset input:")
# with open("tempvec.py", "w") as file:
#     file.truncate(0)
#     file.write("import torch\ninput_vec = torch.tensor([-1, 0, 0, 1, 0, 0, 0, 0, 0, 0.347111, 0.532728, 0, 0.347111, 0.532728, 0, -2*0.347111, -2*0.532728, 0, 35.7071, 35.7071, 35.7071], requires_grad = True)")
#

# [ 1.5204e+00,  7.6287e-01, -7.2413e-02, -1.7299e-01,  4.2930e-01,
#         -9.4932e-02, -6.4748e-01, -3.9216e-01, -3.2655e-02,  7.0786e-01,
#          5.1624e-01, -1.6740e-04,  2.0788e-01,  3.1259e-01, -2.9388e-02,
#         -8.1860e-01, -7.8648e-01,  2.8100e-02,  3.5703e+01,  3.5734e+01,
#          3.5692e+01]
#
# [ 1.5305e+00,  7.2942e-01, -7.9352e-02, -1.7411e-01,  4.6347e-01,
#         -7.3587e-02, -6.5644e-01, -3.9288e-01, -4.7061e-02,  6.4761e-01,
#          5.2369e-01, -6.4135e-03,  1.9012e-01,  2.8642e-01, -7.6145e-03,
#         -7.9527e-01, -7.8276e-01,  1.3103e-02,  3.5702e+01,  3.5733e+01,
#          3.5693e+01]