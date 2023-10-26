
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # client = Client(0, dataset_name, model_name, args, None, device)
    # trainset, testset = client.load_dataset(transform=trans)
    # client.trainloader = client.trainloader if torch.cuda.is_available() else torch.utils.data.DataLoader(CustomSubset(trainset, range(100)), batch_size=16, shuffle=False) 
    # client.train()
    # # client.load_state('results/checkpoints/0_AlexNet_cifar10_.pth')
    # client.eval()
    # client.draw_curve()

# 服务端实现后，可采用num_client=1代替