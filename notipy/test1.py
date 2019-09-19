from notipy.notipy import NotiPy

if __name__ == "__main__":
    def test_fcn(args):
        x, = args;
        for i in range(x):
            print(i)
    sm = "Hi,\nYour function run is complete.\n"
    fcn = test_fcn
    args = (1000,)
    notifier = NotiPy(fcn,args,sm,"asw42695@gmail.com",timeout=None)
    notifier.run()
