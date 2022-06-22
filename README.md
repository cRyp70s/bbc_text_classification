# BBC TEXT CLASSIFICATION

Demo application to classify news articles into one of: 
business, entertainment, politics, sport, tech classes using an LSTM model with pytorch.

Model was train on the [BBC FULL TEXT CLASSIFICATION DATASET](https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification).

    Usage: app.py [OPTIONS] MODE:{cli|gradio} [TEXT]

    Arguments:
      MODE:{cli|gradio}  Mode to run in. cli to run in
                         command             line or
                         gradio to launch a gradio app
                         [required]
      [TEXT]             Text to classify.        Can
                         only be used in cli mode
    
    Options:
      --share / --no-share  share option for gradio
                            launch  [default: no-
                            share]
      --install-completion  Install completion for the
                            current shell.
      --show-completion     Show completion for the
                            current shell, to copy it
                            or customize the
                            installation.
      --help                Show this message and
                            exit.