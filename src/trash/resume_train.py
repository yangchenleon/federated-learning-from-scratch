def train(self, startckpt=None, save=True):
    self.model.train()
    if startckpt is not None:
        self.load_state(startckpt)
        with open(curve_dir+self.file_midname+f"_{self.trained_epoch}.json", 'r') as f:
            self.learn_curve = json.load(f)
    for epoch in (range(self.trained_epoch, self.args.num_epochs)): # tqdm optional
# 在FL环境中貌似是一个伪需求，客户凭什么有自己从文件中load一些，pFl没准可以，不过可能不是这样搞的
# 以及本地的curve是废的，挺鸡肋的，都先放这吧，有用再找出来
            for X, y
            self.learn_curve.append((ls/num_sample, acc/num_sample*100))

def save_state(self):
    torch.save({ }, state_dir+self.file_midname+f'_{self.args.num_epochs}.pth')

    with open(curve_dir+self.file_midname+f'_{self.args.num_epochs}.json', 'w') as f:
        json.dump(self.learn_curve, f)


def draw_curve(self):
    with open(curve_dir+self.file_midname+f'_{self.args.num_epochs}.json', 'r') as f:
        self.learn_curve = json.load(f)
    train_loss, train_acc = zip(*self.learn_curve)
    fig, ax1 = plt.subplots()
    ax1.plot(train_loss, 'r-', label='train_loss')
    ax1.set_xlabel('X')
    ax1.set_ylabel('loss', color='r')
    ax1.tick_params('y', colors='r')
    ax2 = ax1.twinx()

    # 绘制第二个y轴对应的数据
    ax2.plot(train_acc, 'b-', label='train_acc')
    ax2.set_ylabel('accuracy', color='b')
    ax2.tick_params('y', colors='b')

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines])
    plt.savefig(figure_dir + self.file_midname+f'_{self.args.num_epochs}.jpg')
    plt.show()

