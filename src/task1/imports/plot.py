import matplotlib.pyplot as plt



def plot_scenario(robots_pos, targets_pos_real, est_targets_dists, est_targets_pos=None):
    for i in range(len(robots_pos)):
        plt.plot(robots_pos[i][0], robots_pos[i][1], "s", color="tab:blue")

    for i in range(len(targets_pos_real)):
        plt.plot(targets_pos_real[i][0], targets_pos_real[i][1], "x", color="tab:red", label="One Piece")

    for i in range(len(est_targets_dists)):
        for j in range(len(est_targets_dists[i])):
            plt.gca().add_patch(
                plt.Circle(robots_pos[i], est_targets_dists[i,j], color='g', fill=False)
            )
            
    if est_targets_pos is not None:
        for i in range(len(est_targets_pos)):
            for j in range(len(est_targets_pos[i])):
                plt.plot(est_targets_pos[i,j,0], est_targets_pos[i,j,1], "o", color="tab:orange", alpha=0.75, label=f"{i}-Maybe One Piece?")


    plt.axis("scaled")
    plt.legend(bbox_to_anchor=(1, 1))