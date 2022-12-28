
class BitList:

    def __init__(self, fileName):
        self.tags = []
        self.values = []
        self.n = 20
        self.fileName = fileName
        self.mistakes = 0
        self.weights = [1] * 20
        file1 = open(fileName)
        lines = file1.readlines()
        for line in lines:
            listBit = [int(c) for c in line.replace(' ', '').replace('\n', '')]
            self.values.append(listBit[:20])
            self.tags.append(listBit[len(listBit) - 1])
        self.algo()

    def algo(self):
        tag = 0
        run = True
        str = ''
        while (run):
            for x in range(len(self.values)):
                str += "row: {row}\n".format(row=self.values[x])
                if (x == 0):
                    checks = 0
                sum = 0
                for bit in range(len(self.values[x])):
                    sum += self.weights[bit] * self.values[x][bit]
                if (sum >= self.n):
                    str += "The sum is greater than or equal to n, that's why we will assign a tag of 1. [n: {n}, sum: {sum}]\n".format(
                        n=self.n, sum=sum)
                    tag = 1
                else:
                    str += "The sum isn't greater than or equal to n, that's why we will assign a tag of 0. [n: {n}, sum: {sum}]\n".format(
                        n=self.n, sum=sum)
                    tag = 0
                str += 'Line number: {x} | sum = {sum}'.format(x=x, sum=sum)
                if (self.tags[x] == tag):
                    checks += 1
                else:
                    str += "The bit in line at position {x} is not equal to the tag we got, The tag we got is: {tag}\n Therefore we will advance our number of mistakes by 1\n".format(
                        x=x, tag=tag)
                    self.mistakes += 1
                    for bit in range(len(self.values[x])):
                        if (self.values[x][bit] == 1):
                            if (tag == 0):
                                str += "We will multiply the bit in {bit} of weights at position {x} by 2: {value}*2={result} \n Therefore we will advance our number of mistakes by 1\n".format(
                                    bit=bit, x=x, value=self.weights[bit], result=self.weights[bit]*2)
                                self.weights[bit] *= 2
                            else:
                                str += "We will change the weights of the bit in position {bit} in the list of weights from {value} to 0\n".format(
                                    bit=bit, x=x, value=self.weights[bit])

                                self.weights[bit] = 0
                    break

                if (checks == len(self.values) - 1):
                    run = False
        str += '\n\nweights: {weights}'.format(weights=self.getWeights())
        str += '\nweights {mistakes}'.format(mistakes=self.getMistakes())
        path = '{fileName}-output.txt'.format(fileName=self.fileName)
        with open(path, 'w') as f:
            f.write(str)

    def getWeights(self) -> list:
        return self.weights

    def getMistakes(self) -> float:
        return self.mistakes

    def __str__(self):
        str = ''
        run = 0
        for x in range(len(self.values)):
            str += 'values[{index}]: {values} | tag: {tag}\n'.format(
                index=run, values=self.values[x], tag=self.tags[x])
            run += 1
        return str


if __name__ == '__main__':
    fileName = 'winnow_vectors.txt'
    bitList = BitList(fileName)
    print('weights', bitList.getWeights())
    print('mistakes', bitList.getMistakes())
