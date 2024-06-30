# Lottery-Ticket-Hypothesis

- A re-implementation of the paper https://arxiv.org/abs/1803.03635 The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks by Jonathan Frankle and Michael Carbin.
- This implementation focuses only on applying the hypothesis to Fully Connected networks on MNIST .
- Neural network pruning techniques can reduce the parameter counts of trained networks by over 90%, decreasing **storage requirements** and improving **computational performance** of inference **without compromising accuracy** . However, contemporary experience is that the sparse architectures produced by pruning are difficult to train from the start, which would similarly improve training performance.
- We find that a **standard pruning technique** naturally uncovers **subnetworks** whose **initializations** made them capable of training effectively. Based on these results, we articulate the **"lottery ticket hypothesis:"** dense, randomly-initialized, feed-forward networks contain **subnetworks ("winning tickets")** that - when trained in isolation - reach **test accuracy** comparable to the original network in a similar number of iterations. **The winning tickets we find have won the initialization lottery: their connections have initial weights that make training particularly effective.**
- We present an **algorithm** to **identify winning tickets** and a series of **experiments** that support the lottery ticket hypothesis and the importance of these fortuitous initializations. We consistently find winning tickets that are less than **10-20% of the size of several fully-connected and convolutional feed-forward architectures for MNIST and CIFAR10** . Above this size, the winning tickets that we find learn **faster** than the original network and **reach higher test accuracy.**
