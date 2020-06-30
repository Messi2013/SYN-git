       # # show the image visulization by 2.Writing to TensorBoard
        dataiter = iter(train_loader)
        # ipdb.set_trace()
        imagesCam1, images, labels = dataiter.next()
        print('images_cam1_shape', imagesCam1.shape)#(1,8,9,82,46)
        print('images_cam2_shape', images.shape)
        print('labels', labels)
        # tensor dimension images: cam_id x sequence_length x num channels x width x height
        images_color = images[0,:,6:9,:,:] # TODO: make the heatmap a color image
        print('images_color', images_color.shape)
        img_grid = torchvision.utils.make_grid(images_color, pad_value = 1) 
        img_grid_hm = heatmap2image(img_grid)
        print('img_grid', img_grid.shape)

        # matplotlib_imshow(img_grid, one_channel=True)
        img_np = np.transpose(img_grid_hm.numpy(), (1, 2, 0))
        plt.imshow(img_np)
        train_writer.add_image('heatmap_images', img_grid_hm)

        dataiter1 = iter(train_loader)
        # ipdb.set_trace()
        _, images1, labels1 = dataiter1.next()
        print('images', images1.shape)#(1,8,9,82,46)
        # tensor dimension images: cam_id x sequence_length x num channels x width x height
        images_color1 = images1[0,:,0:3,:,:] # TODO: make the heatmap a color image
        print('images_color', images_color1.shape)
        img_grid1 = torchvision.utils.make_grid(images_color1, pad_value = 1) 
        img_grid_hm1 = heatmap2image(img_grid1)
        print('img_grid', img_grid1.shape)