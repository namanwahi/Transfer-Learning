from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.autograd import Variable

def train_model(model,dataloader, op_params=None, loss_fn=CrossEntropyLoss, epochs=12, initial_learning_rate =1e-2):
    if (op_params == None):
        optimizer = Adam(model.parameters(), lr=initial_learning_rate)
    else:
        optimizer = Adam(op_params, lr=initial_learning_rate)

    current_loss = 0
    for i in range(epochs):
        for x_b, y_b in dataloader:

            x_b, y_b = Variable(x_b), Variable(y_b)

            output = model(x_b)
            loss = loss_fn(output, y_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = loss.data[0]
            print(current_loss)

    
    
