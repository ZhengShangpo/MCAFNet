
def feature_visualization_all(x, save_dir=Path('featuremap/exp')):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """

    batch, channels, height, width = x.shape  # batch, channels, height, width
    if height > 1 and width > 1:
        f = save_dir / f"features.png"  # filename
        img = x[0].cpu().transpose(0, 1).sum(1).detach().numpy()
        plt.imsave(f, img)


# def feature_visualization1(x, m_i, m_type,i,  n=32, save_dir=Path('featuremap/exp')):
#     """
#     x:              输入即可视化的Tensor
#     module_type:    Module type 用于命名区分各层特征图
#     stage:          Module stage within model 用于命名区分各层特征图
#     n:              Maximum number of feature maps to plot 可视化的通道个数（通道数太多不可能全部可视化）
#     save_dir:       Directory to save results 特征图的保存路径
#     """
#     batch, channels, height, width = x.shape  # batch, channels, height, width
#     if height > 1 and width > 1:
#         # 文件的命名格式 层名+层的索引
#         f = save_dir / f"{m_i}+{m_type}{i}features.png"  # filename
#         # 按通道数拆分Tensor
#         # 进行逐通道的可视化
#         blocks = torch.chunk(x[0], channels, dim=0)  # select batch index 0, block by channels
#         n = min(n, channels)  # number of plots
#         fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
#         ax = ax.ravel()
#         plt.subplots_adjust(wspace=0.05, hspace=0.05)
#         for i in range(n):
#             ax[i].imshow(blocks[i].detach().cpu().numpy().squeeze())  # cmap='gray'
#             ax[i].axis('off')
#
#         plt.savefig(f, dpi=300, bbox_inches='tight')
#         plt.close()
#         np.save(str(f.with_suffix('.npy')), x[0].detach().cpu().numpy())  # npy save

def feature_visualization1(x, m_i, m_type,i, exp, n=32, save_dir='featuremap',figsize=(640, 640)):
    """
    x:              输入即可视化的Tensor
    module_type:    Module type 用于命名区分各层特征图
    stage:          Module stage within model 用于命名区分各层特征图
    n:              Maximum number of feature maps to plot 可视化的通道个数（通道数太多不可能全部可视化）
    save_dir:       Directory to save results 特征图的保存路径
    """
    batch, channels, height, width = x.shape  # batch, channels, height, width
    if height > 1 and width > 1:
        # 文件的命名格式 层名+层的索引
        batch, channels, height, width = x.shape  # batch, channels, height, width
        save_dir1 = os.path.join(save_dir,exp)
        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1)
        if height > 1 and width > 1:
            f = save_dir1+'/'+f"{m_i}+{m_type}+{i} features.png"  # filename
            img = x[0].cpu().transpose(0, 1).sum(1).detach().numpy()
            plt.figure(figsize=figsize)
            plt.imsave(f, img, cmap='viridis',dpi=600)


def feature_visualization(x, m_i, m_type, exp, n=32, save_dir='featuremap',figsize=(640, 640)):
    """
    x:              输入即可视化的Tensor
    module_type:    Module type 用于命名区分各层特征图
    stage:          Module stage within model 用于命名区分各层特征图
    n:              Maximum number of feature maps to plot 可视化的通道个数（通道数太多不可能全部可视化）
    save_dir:       Directory to save results 特征图的保存路径
    """
    batch, channels, height, width = x.shape  # batch, channels, height, width
    if height > 1 and width > 1:
        # 文件的命名格式 层名+层的索引
        batch, channels, height, width = x.shape  # batch, channels, height, width
        save_dir1 = os.path.join(save_dir,exp)
        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1)
        if height > 1 and width > 1:
            f = save_dir1+'/'+f"{m_i}+{m_type} features.png"  # filename
            img = x[0].cpu().transpose(0, 1).sum(1).detach().numpy()
            plt.imsave(f, img, cmap='viridis',dpi=600)