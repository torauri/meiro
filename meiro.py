import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer.training import extensions

class MyModel(chainer.Chain):
    def __init__(self):
        super(MyModel, self).__init__(
                l1=L.Linear(3,10),
                l2=L.Linear(10,4)
        )

    def __call__(self, x):

        h = F.relu(self.l1(x))
        y = F.softmax(self.l2(h))

        return y.max()


def action(x,y,a):
    stage = [   [[0,1,1,0],[0,1,1,1],[0,0,0,1],[0,0,1,0],[0,0,1,0]],
                [[1,0,0,0],[1,1,0,0],[0,0,1,1],[1,1,1,0],[1,0,1,1]],
                [[0,1,1,0],[0,1,0,1],[1,1,0,1],[1,0,1,1],[1,0,1,0]],
                [[1,0,1,0],[0,1,1,0],[0,0,1,1],[1,0,0,0],[1,0,1,0]],
                [[1,1,0,0],[1,0,0,1],[1,1,0,0],[0,1,0,1],[0,0,0,0]]]

    ax=x
    ay=y
    if(stage[y][x][a]==1):
        if(a==0):
            ay=ay-1
        elif(a==1):
            ax=ax+1
        elif(a==2):
            ay=ay+1
        elif(a==3):
            ax=ax-1

    return ax,ay

def show_status(x,y):
    status = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    status[y][x]=1
    for s in status:
        print (s)
    return

def meiro():
    x=0
    y=0

    while(x!=4 or y!=4):
        print("0:上　1:右　2:下 3:左")
        a = eval(input('>>'))
        x,y = action(x,y,a)


        show_status(x,y)

    print("ゴール")

def main():
    meiro()

if __name__ == '__main__':
    main()
