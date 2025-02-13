# shape loss
if self.opts.shape_lambda > 0:
    # input_ages = self.aging_loss.extract_ages(y) / 100.
    # target_ages = self.aging_loss.extract_ages(y_hat) / 100.
    # print(target_ages.shape)
    # print(input_ages.shape)
    for i in range(target_ages.size(0)):
        # print(target_ages[i].item())

        if target_ages[i].item() >= input_ages[i].item():  # (都是除100了)
            # target_ages - input_ages 可以分段
            # print('older')

            d, d2 = self.landmark_detector.detect_landmarks(common.tensor2im(y[i]))
            d3, d4 = self.landmark_detector.detect_landmarks(common.tensor2im(y_hat[i]))
            # print(distance , distance2)

            loss_shape = 1 / 2 * (abs(d - d3) + abs(d2 - d4))

            loss_dict[f'loss_shape_{data_type}'] = float(loss_shape)
            loss += loss_shape * self.opts.shape_lambda
        else:
            # 如果年齡由大變小，增加額外的懲罰
            # print('younger')
            d5, d6 = self.landmark_detector.detect_landmarks(common.tensor2im(y[i]))  # common.tensor2im PIL image
            d7, d8 = self.landmark_detector.detect_landmarks(common.tensor2im(y_hat[i]))
            loss_shape = 1 / 2 * (abs(d5 - d7) + abs(d6 - d8))
            penalty = self.opts.penalty_value * abs(input_ages[i].item() - target_ages[i].item())

            loss_dict[f'loss_shape_{data_type}'] = float(loss_shape)
            loss += (loss_shape + penalty) * self.opts.shape_lambda