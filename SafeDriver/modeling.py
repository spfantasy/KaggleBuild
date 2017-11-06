class dataset:
    def __init__(self, data=None, label=None):
        self.X = data
        self.y = label

class Modeling(object):
    @staticmethod
    def loadtrain(path):
        train = dataset(pd.read_csv(path+"train_X"+".csv"),
                   pd.read_csv(path+"train_y"+".csv"))
        return train

    @staticmethod
    def loadtest(path):
        test = dataset(pd.read_csv(path+"test_X"+".csv"),
                   pd.read_csv(path+"test_y"+".csv"))
        return test

    @staticmethod
    def loadcv(path):
        cv = []
        i = 0
        while True:
            try:
                train = dataset(pd.read_csv(path+"train_X_"+str(i)+".csv"),
                        pd.read_csv(path+"train_y_"+str(i)+".csv"))
                valid = dataset(pd.read_csv(path+"valid_X_"+str(i)+".csv"),
                        pd.read_csv(path+"valid_y_"+str(i)+".csv"))
                cv.append([train, valid])
                print("cross validation set %d loaded"%i)
                i += 1
            except:
                return cv


    #I'm a decorator
    @staticmethod
    def metacv(func):
        def wrapper(methodname, cv, test, eval_func):
            testresult = []
            cvresult = []
            for i,data in enumerate(cv):
                print("training CV round %d"%i)
                v, t = func(data[0], data[1], test)
                cvresult.append(v)
                testresult.append(t)
            test.y['target'] = np.mean(testresult,axis = 0)
            test.y.to_csv('./predictions/meta_test/%s.csv'%methodname, index=False, float_format='%.5f')
        return wrapper

    #I'm a decorator
    @staticmethod
    def gridsearchcv():
        pass

    #because the return cv predictions are shuffled under validation sets
    #In X->predictions->y
    #stage2 preditions->y
    #y need to be reorder as by cv[i][1].y(and index killed)
    @staticmethod
    def stacking():
        pass
