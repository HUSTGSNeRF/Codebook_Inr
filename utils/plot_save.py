import numpy as np
import matplotlib.pyplot as plt


def plot_loss(losses_train, exp_title, save_dir):
    """asda

    Args:
        data_dict (_type_): _description_
        exp_title (_type_): _description_
        save_dir (_type_): _description_
    """
    fig, ax1 = plt.subplots()
    losses_train = np.log10(losses_train)
    ax1.plot(losses_train, label="log_loss")
    ax1.set_ylabel("MSE loss-log10")
    ax1.set_xlabel("EPOCHS")
    
    # losses_train = np.log10(losses_train)
    # ax2 = ax1.twinx()
    # ax2.plot(losses_train, label="log_loss", color="yellow")
    
    # max_indx = np.argmax(acc_test)  # min value index
    # ax2.plot(max_indx, acc_test[max_indx], 'ko')
    # show_min = "%0.2f" % (acc_test[max_indx])
    # ax2.annotate(show_min, xy=(max_indx, acc_test[max_indx]),
    #              xytext=(max_indx - 20, acc_test[max_indx] - 0.05))
    #
    # max_indx = np.argmax(totalacc_test)  # min value index
    # ax2.plot(max_indx, totalacc_test[max_indx], 'ko')
    # # show_min='('+str(min_indx)+' '+str(losses[min_indx])+')'
    # show_min = "%0.2f" % (totalacc_test[max_indx])
    # ax2.annotate(show_min, xy=(max_indx, totalacc_test[max_indx]),
    #              xytext=(max_indx - 20, totalacc_test[max_indx] - 0.05))
    plt.title(exp_title)
    fig.legend(loc=5, bbox_to_anchor=(0.85, 0.5))
    plt.savefig(save_dir)


def plot_save(data_dict, exp_title, save_dir):
    """asda

    Args:
        data_dict (_type_): _description_
        exp_title (_type_): _description_
        save_dir (_type_): _description_
    """
    losses_train = data_dict["losses_train"]
    losses_test = data_dict["losses_test"]
    acc_test = data_dict["acc_test"]
    totalacc_test = data_dict["totalacc_test"]

    fig, ax1 = plt.subplots()
    ax1.plot(losses_train, label="train_loss", color="blue")
    ax1.plot(losses_test, label="test_loss", color="red")
    if np.min(losses_train) < 0:
        ax1.set_ylabel("lg(MSE)")
    else:
        ax1.set_ylabel("MSE")
    ax1.set_xlabel("EPOCHS")
    ax2 = ax1.twinx()
    ax2.plot(acc_test, label="unit acc", color="green")
    ax2.plot(totalacc_test, label="sample acc", color="yellow")
    # max_indx = np.argmax(acc_test)  # min value index
    # ax2.plot(max_indx, acc_test[max_indx], 'ko')
    # show_min = "%0.2f" % (acc_test[max_indx])
    # ax2.annotate(show_min, xy=(max_indx, acc_test[max_indx]),
    #              xytext=(max_indx - 20, acc_test[max_indx] - 0.05))
    #
    # max_indx = np.argmax(totalacc_test)  # min value index
    # ax2.plot(max_indx, totalacc_test[max_indx], 'ko')
    # # show_min='('+str(min_indx)+' '+str(losses[min_indx])+')'
    # show_min = "%0.2f" % (totalacc_test[max_indx])
    # ax2.annotate(show_min, xy=(max_indx, totalacc_test[max_indx]),
    #              xytext=(max_indx - 20, totalacc_test[max_indx] - 0.05))
    plt.title(exp_title)
    fig.legend(loc=5, bbox_to_anchor=(0.85, 0.5))
    plt.savefig(save_dir, dpi=600)


# 设置图像的标题为 exp_title。
# 添加图例到图像中，其中 loc=5 表示图例位于右下角。
# bbox_to_anchor 参数用于指定图例的位置，(0.85, 0.5) 表示以图像右边缘的 85% 和底边缘的 50% 作为参考点放置图例。
# 将生成的图像保存到指定的文件夹路径 save_dir 中，并设置保存图像的分辨率为 600 dpi。

