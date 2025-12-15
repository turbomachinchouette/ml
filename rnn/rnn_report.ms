.\" Centered caption for figure.  Assumes previous .KS
.de CE
.  ce 1
Figure \\n[H1]-\\$1
.  sp 1
.  KE
..
.de LI
.  ft R
.  ce 1
Listing \\n[H1]-\\$1
.  sp 1
.  KE
..

.EQ
delim $$
.EN

.P1
.DA
.TL
A Report on Recursive Neural Nets Applied to SIMGAIT
.AU
Gabriel Rolland
.AI
HES-SO University of Applied Sciences and Arts Western Switzerland
.AB
In this paper, we analyze RNN applied to the SIMGAIT dataset.
Recurrent neural nets are neural nets that 'remember'
all the previous layers in the forward pass.
This memory is useful to capture dependencies in the parameters.
But as all recursive methods in computational methods, it's expensive
because you have to keep in memory most of the recursive stack.
This problem is what has led to the adoption of Transformers.
.[
vaswani transformers
.]

.AE
.NH
Steps
.PP
This was the approach:
.IP \[bu]
We implement a RNN in python
.IP \[bu]
We test the RNN against two variants: LSTM and GRU.
.IP \[bu]
We compare SGD and ADAM and how well each applies to each model.
.IP \[bu]
We test the RNN with two different activations Sigmoid and ReLU.
.IP \[bu]
We test each activation with and without ad hoc optimization.
.IP \[bu]
We see which number of epochs is necessary to achieve convergence.
.IP \[bu]
We finally compare RNN with an attention mechanism like Transformers (or Ptr-net.)

.NH
Theory and Implementation of RNN
.PP
A recurrent neural net
.I remembers
a memory by passing a hidden state to each successive layers.
We draw two figures to compare the RNN architecture with a fully connected feed-forward MLP.
.KS
.PS
move
box "Input" at 0, 0 width 0.8 height 0.3 ;
box "Hidden Layer" at 0, .5 width 1 height 0.3  dashed;
box "Output" at 0, 1 width 0.8 height 0.3 ;
arrow at 0, 0.15 to 0, 0.3
arrow at 0, 0.65 to 0, 0.8
.PE
.CE "1: A feed-forward MLP"

.KS
.PS
move
box "Input" at 0, 0 width 0.8 height 0.3 ;
box "Hidden Layer" at 0, .5 width 1 height 0.3  dashed;
box "Output" at 0, 1 width 0.8 height 0.3 ;

arrow at 0, 0.15 to 0, 0.3
arrow at 0, 0.65 to 0, 0.8

right; move 0.2; down; move 0.1
line up 0.05 then right then down 0.25 then left 0.15 ->;
right; move 0.3;
"$\fBH\fR sup t$"

.PE
.CE "2: RNN where \fBH\fR denotes Hidden State"
.LP
The question arises of how do we backpropagate errors in this type of architecture?
Well, we recursively sum all losses through time.
This is called
.B BPTT
(Backpropagation through time).
As shown in the following equations:
.[
rnn schmidt
.]
.EQ (1)
{partial L} over {partial W sub hh} = sum sub {iota T} {partial l sub t} over
{partial O sub t}  {partial O sub t} over {partial phi sub o} W sub ho
sum sub {iota t} ( W sup T sub hh ) sup {t - iota} H sub iota 
.EN
.EQ (2)
{partial L} over {partial W sub xh} = sum sub {iota T} {partial l sub t} over
{partial O sub t}  {partial O sub t} over {partial phi sub o} W sub ho
sum sub {iota t} ( W sup T sub hh ) sup {t - iota} X sub iota 
.EN

Where:
.IP \[bu]
$iota$ is the index operator we borrow from APL
.[
iverson apl
.]
(e. g. $iota 5 = 1, 2, 3, 4, 5$, $iota$ with no arguments denotes current index).
.IP \[bu]
O is our Output.
.IP \[bu]
W is the weights of hidden layer H w.r.t. input or output.
.IP \[bu]
X is the input.

.SH
Implementation
.PP
We implement the following models in pytorch:
.KS
.ft C
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        # Define your new architecture here

        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Define the forward pass
        x = x.view(x.shape[0], x.shape[1], 1)
        out = self.rnn(x)[0]
        out = out[:, -1, :]
        out = self.output_layers(out)
        return out
# With those hyperparameters
epochs = 10
batch_size = 10
learning_rate = 1e-2
eval_every = 1  # number of epochs between evaluations

# Optimizer SGD and loss BCE
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss(reduction="sum")
.LI "1: vanilla rnn"
.LP
We chose to make the model deliberately as simple as possible and start from there.
Since we have a binary output, our loss is given by Binary Cross Entropy and we
chose the sigmoid as the last activation. Here is the result over 10 epochs:
.KS
.PSPIC "./simpleRNN.eps"
.CE "3: validation, test and train loss for a simple RNN"

.KS
.ft C
# larger LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Define your new architecture here

        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss(reduction="sum")
.LI "2: LSTM"
.KS
.PSPIC "./largerLSTM.eps"
.CE "5: Loss for LSTM"
.LP
We see a much sharper decrease in the train loss. We aimed for a
bigger number of epochs but we see almost no further fit beyond a
few epochs.

.KS
.ft C
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        # Define your new architecture here

        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

.LI "3:  GRU"
.KS
.PSPIC "./GRU.eps"
.CE "5: 10 epochs with GRU"
.LP
Conversely, we observe almost no fitting with GRU. The results are
very similar to a vanilla RNN optimized with SGD.
This sort of stalling in Fig 2-3 and 2-5 is to be expected because we did take any measure
to prevent exploding or vanishing gradients like batch normalization
or Xavier Glorot initialization. Our learning rate was probably too big too,
we usually see values of $1 e sup -6$.
.NH
Predicting gait

.PP
We move to not only predict labels, but to predict further points.


.NH
Conclusions




