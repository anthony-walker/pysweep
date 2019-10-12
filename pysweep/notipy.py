
from collections.abc import Iterable
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import signal

class timeout:
    def __init__(self, seconds=1, error_message='Your function has timed out.'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class NotiPy(object):
    """Use this class to execute a function and notify via email when it has completed."""
    def __init__(self,fcn,fcn_args,success_message,receiver,rank=0,master_rank=0,timeout=None):
        """Initializer."""
        self.fcn = fcn
        self.fcn_args = fcn_args
        self.timeout = timeout
        fromaddr = "you@gmail.com"
        toaddr = "target@example.com"
        self.smtp ='smtp.gmail.com'
        self.server = smtplib.SMTP(self.smtp, 587)
        #Next, log in to the server
        self.email = "dev.notipy"
        self.receiver = receiver
        #Message
        self.success = MIMEMultipart()
        self.success['From'] = self.email+"@gmail.com"
        self.success['To'] = self.receiver
        self.success['Subject'] = "Code Completion."
        self.sm = success_message
        self.rank = rank
        self.master = master_rank

    def send(self):
        """This is the server login function."""
        self.server.connect(self.smtp,587)
        self.server.ehlo()
        self.server.starttls()
        self.server.ehlo()
        self.server.login(self.email, "ewxzkrhkefk12")

        self.server.sendmail(self.email+"@gmail.com",self.receiver,self.success.as_string())
        self.server.quit()

    def run(self):
        """Call run to execute the given function and arguments."""
        if self.rank == self.master:
            if self.timeout is None:
                try:
                    self.rets = self.fcn(self.fcn_args)
                    if isinstance(self.rets,Iterable):
                        self.sm+="\nReturns:\n"
                        for i,arg in enumerate(self.rets):
                            self.sm += str(arg)+"\n"
                    elif self.rets is not None:
                        self.sm+="\nReturns:\n"
                        self.sm += str(self.rets)
                except Exception as e:
                    self.sm += "Code execution failure:\n"+str(e)
            else:
                #Attempt to solve with timeout
                try:
                    with timeout(seconds=self.timeout):
                        self.rets = self.fcn(self.fcn_args)
                        if isinstance(self.rets,Iterable):
                            self.sm+="\nReturns:\n"
                            for i,arg in enumerate(self.rets):
                                self.sm += str(arg)+"\n"
                        elif self.rets is not None:
                            self.sm+="\nReturns:\n"
                            self.sm += str(self.rets)
                except Exception as e:
                    self.sm += "\nCode execution failure:\n"+str(e)
            #Attempt to send email
            try:
                ts = MIMEText(self.sm)
                self.success.attach(ts)
                self.send()
            except Exception as e:
                print(e)

        else:
            if self.timeout is None:
                self.rets = self.fcn(self.fcn_args)
            else:
                with timeout(seconds=self.timeout):
                    self.rets = self.fcn(self.fcn_args)


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
