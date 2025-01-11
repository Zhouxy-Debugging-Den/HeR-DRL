from torch.utils.data import Dataset


class ReplayMemory(Dataset):
    """
    The stored content is a list, which contains state, value, done, reward, next_state
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    def push(self, item):
        # replace old experience with new experience
        if len(self.memory) < self.position + 1:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()


class ReplayMemory_hetero(Dataset):
    """
    What is the data stored in memory?


    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    def push(self, item):
        if len(self.memory) < self.position + 1:
            # If it's not full, keep filling it.
            self.memory.append(item)
        else:
            # Replace old experience
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()
