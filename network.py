
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data, y_test):
        # sample dimension first
        samples = len(input_data)
        result = []
        err = 0

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
            err += self.loss(y_test[i], output)

        return result, err / samples

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)
        last_err = 1
        last_err1 = 1

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            if (last_err - err < 0.000001 and last_err - err > -0.000001 and last_err1 - last_err < 0.0000001 and last_err1 - last_err > -0.0000001):
                break
            # print('epoch %d/%d   error=%f   min=%f' % (i+1, epochs, err, last_err - err))
            last_err1 = last_err
            last_err = err
            # if err < 0.00003:
            #     break
        return err