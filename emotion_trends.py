import math


import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as stats


class Trends:
    def __init__(self, visual_type='TIBAV', data_type='correct'):
        self.visual_type = visual_type
        if data_type == 'correct':
            output, labels = self.extract_correct_predictions()
        else:
            output, labels = self.extract_wrong_predictions()
        self.emotion_dict = self.split_data(output, labels)
        self.labels = {0: 'happy', 1: 'sad', 2: 'anger', 3: 'fear', 4: 'disgust', 5: 'neutral'}

    def extract_correct_predictions(self):
        with open("CREMA_D/Pickle dump/" + self.visual_type + "/correct_output", "rb") as f:
            correct_output = pickle.load(f)

        with open("CREMA_D/Pickle dump/" + self.visual_type + "/correct_labels", "rb") as f:
            correct_labels = pickle.load(f)

        return correct_output, correct_labels

    def extract_wrong_predictions(self):
        with open("CREMA_D/Pickle dump/" + self.visual_type + "/wrong_output", "rb") as f:
            wrong_output = pickle.load(f)

        with open("CREMA_D/Pickle dump/" + self.visual_type + "/wrong_labels", "rb") as f:
            wrong_labels = pickle.load(f)

        return wrong_output, wrong_labels


    def split_data(self, attn, labels_arr, attn_type='spc_time'):
        assert attn.shape[0] == labels_arr.shape[0], "Shapes of attention and labels do not match."
        if attn_type == 'spc_time':
            attn = attn[:, 0, :, :]
        elif attn_type == 'time1':
            attn = attn[:, 1, :, :]
        elif attn_type == 'time2':
            attn = attn[:, 2, :, :]
        labels_dict = defaultdict(list)
        counter = 0
        for k, v in list(zip(attn, labels_arr)):
            labels_dict[v].append(k)
            counter += 1

        return labels_dict

    def find_max_min(self):
        vmin = 0
        vmax = 0
        for emotion in self.emotion_dict:
            average = np.mean(np.stack(self.emotion_dict[emotion]), axis=0)
            max_val = np.max(average)
            min_val = np.min(average)
            if max_val > vmax:
                vmax = max_val
            if min_val < vmin:
                vmin = min_val

        return vmin, vmax


    # plot heatmaps of each emotion averaged
    def average_heatmap(self, reduced=False):
        for emotion in self.emotion_dict:
            print(self.labels[emotion])
            average = np.mean(np.stack(self.emotion_dict[emotion]), axis=0)
            average = average.reshape((8, 14, 14))
            for i, frame in enumerate(average):
                if reduced:
                    frame = self.reduce_array(frame)
                plt.subplot(2, 4, i+1)
                plt.imshow(frame, cmap='hot', interpolation='nearest')
                plt.title(self.labels[emotion] + str(i))
            plt.show()

    # plot line graph of the patches for each time and each emotion
    def time_line_plt(self, reduced=False):
        for emotion in self.emotion_dict:
            print(self.labels[emotion])
            average = np.mean(np.stack(self.emotion_dict[emotion]), axis=0)
            for i, frame in enumerate(average):
                if reduced:
                    frame = frame.reshape((14,14))
                    frame = self.reduce_array(frame).reshape(49)
                    plt.plot(range(49), frame, label=str(i))
                else:
                    plt.plot(range(196), frame, label=str(i))
            plt.title(self.labels[emotion])
            plt.legend()
            plt.show()

    def reduce_array(self, frame):
        frame_x_red = np.zeros((7, 14))
        frame_red = np.zeros((7, 7))
        for i in range(7):
            for j in range(14):
                frame_x_red[i][j] = (frame[2*i][j] + frame[(2*i)+1][j])/2
        for i in range(7):
            for j in range(7):
                frame_red[i][j] = (frame_x_red[i][2*j] + frame_x_red[i][(2*j)+1])/2

        return frame_red

    def row_line_plt(self, reduced=False):
        for emotion in self.emotion_dict:
            print(self.labels[emotion])
            average = np.mean(np.stack(self.emotion_dict[emotion]), axis=0)
            for i, frame in enumerate(average):
                if reduced:
                    frame = frame.reshape((14,14))
                    frame = self.reduce_array(frame).reshape(49)
                    plt.plot(range(49), frame, label=str(i))
                else:
                    frame = np.mean(frame.reshape((14,14)), axis=1)
                    plt.plot(range(14), frame, label=str(i))
            plt.title(self.labels[emotion])
            plt.legend()
            plt.show()

    def row_line_anim(self):
        for emotion in self.emotion_dict:
            print(self.labels[emotion])
            average = np.mean(np.stack(self.emotion_dict[emotion]), axis=0)
            fig, ax = plt.subplots()
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 0.0015)
            line, = ax.plot(range(14), np.mean(average[0].reshape((14,14)), axis=1))
            plt.title(self.labels[emotion] + ' ' + str(1))
            def animate_frame(i):
                y_data = np.mean(average[i].reshape((14,14)), axis=1)
                line.set_ydata(y_data)
                line.set_xdata(range(14))
                plt.title(self.labels[emotion] + ' ' + str(i+1))
                return line

            animation = FuncAnimation(fig, func=animate_frame, frames=np.arange(0, 8, 1), interval=1000)
            x_labels = ['Forehead', 'Eyes', 'Nose', 'Mouth', 'Chin']
            plt.xlabel('Patches')
            ax.tick_params(axis='x', which='minor', bottom=False)
            ax.set_xticks(np.arange(0, 15, step=3))
            ax.set_xticks(np.arange(1.5, 16.5, step=3), minor=True)
            ax.set_xticklabels(x_labels, minor=True)
            ax.set_xticklabels([])
            plt.ylabel('Attention')
            plt.show()

    def average_heatmap_anim(self, to_scale=True):
        if to_scale:
            vmin, vmax = self.find_max_min()
        for emotion in self.emotion_dict:
            print(self.labels[emotion])
            average = np.mean(np.stack(self.emotion_dict[emotion]), axis=0)
            average = average.reshape((8, 14, 14))
            if to_scale is False:
                vmin = np.min(average)
                vmax = np.max(average)
            fig, ax = plt.subplots()
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '5%', '5%')
            im = ax.imshow(average[0], vmin=vmin, vmax=vmax, cmap='hot', interpolation='nearest')
            cb = fig.colorbar(im, cax=cax)
            plt.title(self.labels[emotion] + ' ' + str(1))
            def animate_frame(i):
                im = ax.imshow(average[i], vmin=vmin, vmax=vmax, cmap='hot', interpolation='nearest')
                plt.title(self.labels[emotion] + ' ' + str(i+1))
                return im

            animation = FuncAnimation(fig, func=animate_frame, frames=8, interval=1000)
            plt.show()

    def gaussian_patch_plots(self):
        for emotion in self.emotion_dict:
            print(self.labels[emotion])
            patch = np.stack(self.emotion_dict[emotion])[:, 0, 0]
            print(patch.shape)
            mu = np.mean(patch)
            var = np.var(patch)
            sigma = math.sqrt(var)
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma))
            plt.show()
            exit(1)

    def most_attn_frame(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        for emotion in self.emotion_dict:
            print(self.labels[emotion])
            ave_samples = np.mean(np.stack(self.emotion_dict[emotion]), axis=0)
            ave_frames = np.mean(ave_samples, axis=1)
            std_frames = np.std(ave_samples, axis=1)/ave_samples.shape[1]
            # std_frames = np.std(ave_samples, axis=1)
            plt.plot(range(8), ave_frames, label=self.labels[emotion])
            plt.fill_between(range(8), ave_frames - std_frames, ave_frames + std_frames, alpha=0.2)
        plt.ylabel("Average attn across all patches")
        plt.xlabel("Frame")
        plt.title("Most attentive frames")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


if __name__ == '__main__':
    import matplotlib; matplotlib.use("TkAgg")
    compute_trends = Trends()
    # compute_trends.average_heatmap()
    compute_trends.most_attn_frame()


