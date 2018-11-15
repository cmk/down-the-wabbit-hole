# Configuring Handles

Note:


## Configuring a TF Handle 
```haskell
module Policy.Model.TF.Simple.Config where

import qualified TensorFlow.Core as TF
                              
data Simple = Simple { unSimple :: TFModelConfig } -- 1

type Float' = TF.TensorData Float                  -- 2

class ModelConfig SimpleTF where
    type SInput Simple = Float'            
    type LInput Simple = (Float', Float')          -- 3
    type Output Simple = Float             
    type Error  Simple = Float

    createModelHandle = TF.build . mkHandle        -- 4
    ...

```
Note: let's configure a single variable regression model y = mx + b
- basic Framework config (e.g. logging verbosity etc)
- i'm opting to feed it data from within TF. 
- training input consists of (x,y) pairs
- my model will supply its own mkHandle and TF.build will render th


# 
```haskell
-- https://github.com/tensorflow/haskell
TF.build :: Build a -> SessionT m a

newtype BuildT m a = BuildT (StateT GraphState m a)

type Build = BuildT Identity
```
```haskell
newtype SessionT m a = Session (ReaderT SS (BuildT m) a)
  deriving (Monad, MonadIO, MonadMask)

type Session = SessionT IO
```
<!-- .element: class="fragment" -->
Note:
- the Build monad defines the computation. Implemented as a state monad over the dependency graph. It doesn’t compute anything 
- the Session monad allows to execute graphs or part of graphs. It allocates resources (on one or more machines) for that and holds the actual values of intermediate results and variables. Implemented as a ReaderT over a session state.


# 
```haskell
type SimpleHandle = ModelHandle' Simple TF.Session

mkHandle :: TF.Build SimpleHandle                  -- 1
mkHandle = do
  let batchSize = -1                               -- 2
  x <- TF.placeholder [batchSize]
  y <- TF.placeholder [batchSize]
  m <- TF.initializedVariable 0
  b <- TF.initializedVariable 0

  let yhat = x * TF.readValue m + TF.readValue b
      loss = TF.square (yhat - y)

      score' = TF.render @TF.Build @Float yhat     -- 3
      error' = TF.render @TF.Build @Float loss

      sgd  = TF.gradientDescent 0.001 

  learn' <- TF.minimizeWith sgd loss [m, b]        -- 4
  return ModelHandle { ... }                       -- 5
```

Note: here's a simple single variable regression model y = mx + b
- Build is a State monad over the graph nodes. I'm simply creating the TF graph here. this requires no data dependencies
- want variable input so i dont' couple my model definition to training details
- render the output tensors representing the dependant variable and error. render fixes the name, scope, device and control inputs from the MonadBuild context. 
- configure my optimization step. standard sgd is fine for something this simple. for more complex models i would probably want to store the opt hyperparameters in the configuration type.
- ran out of room but the creating the model handle is simply a matter of attaching the rendered variables to the x and y data feeds


## Configuring a VW Handle
```
# Test 22: matrix factorization -- training
vw  -k -d train-sets/ml100k_small_train -b 16 -q ui --rank 10 \
    --l2 2e-6 --learning_rate 0.05 --passes 2 \
    --decay_learning_rate 0.97 --power_t 0 -f models/movielens.reg \
    -c --loss_function classic --holdout_off
      train-sets/ref/ml100k_small.stdout
      train-sets/ref/ml100k_small.stderr
```
<!-- .element: class="fragment" -->
```
# Test 128: cb_explore_adf with epsilon-greedy exploration
vw  --cb_explore_adf --epsilon 0.1 -d train-sets/cb_test.ldf \
    --noconstant -p cbe_adf_epsilon.predict
      train-sets/ref/cbe_adf_epsilon.stderr
      pred-sets/ref/cbe_adf_epsilon.predict
```
<!-- .element: class="fragment" -->

Note:
- now lets look at a vw instance. is is similar to TF in that both are designed around SGD and its relatives, both are C++. 
- vw is in many ways opposite to TF. CLI only, can't make up new models.
- here we are using sgd to train a collaborative filtering model
- most relevant for us is the contextual bandit use case with action-dependant features (adf).
- we're reading in a data file cb_test.ldf and training an adf model w/ sgd
- we're also generating predictions and error / stats logs as we go
- the aim of the rest of this talk is to show how we productionize this exact use case
- i'm going to start at the FFI level and work my way up


#  
```haskell
module Policy.Framework.VW.FFI

data VW'Session                                    -- 1
data Session = Session
    { sessionPtr  :: !(Ptr VW'Session)             -- 2
    , predictType :: !PredictType                  -- 3
    , labelType   :: !LabelType 
    , options     :: !ByteString                   -- 4
    } deriving (Show, Eq)
```
```haskell
data VW'Example
newtype Example = Example (Ptr VW'Example)
```
<!-- .element: class="fragment" -->
Note: 
- we have our own vw version of a session. basically a pointer to a location in C++ memory, which we model with an opaque pointer alias. pretty common trick that the TF bindings use as well.
  main difference is that we cant add much beyond that
- label and predict are large sum types of maybe a dozen options each, depending on the configuration options 
- so the main things that go into a session are input type, output type, and a string w/ command line options flags, which is unfortunate but VW's internal representation of options is just a bytestring
- finally instead of a tensor we have an example


#   
```haskell
module Policy.Framework.VW.FFI where

createSession :: ByteString -> IO Session          
deleteSession :: Session -> IO ()

createExample :: Session -> ByteString -> IO Example
deleteExample :: Session -> Example -> IO ()         
```
```haskell
learn :: Session -> Example -> IO ()               
predict :: Session -> Example -> IO () 
getStatistics :: Session -> IO Statistics            
```
<!-- .element: class="fragment" -->

Note: in contrast to the Session Monad in TF, the API for a session in VW is pretty limited 
- just showing the types here, explanation of lower level code is in appendix B & C
- learn and predict have the same type sig. still have to marshall data in to the Example pointer.
- can read some basic statistics out, we will use this in the error function of our model handle
- can read data out of the Example pointer.  have to make sure the input / output types agree w/ the configuration options
- finally can create and destroy examples and sessions


#  
```haskell
import Control.Exception (bracket)

withSession :: ByteString -> (Session -> IO a) -> IO a
withSession args = 
  bracket (createSession args) deleteSession

withExample 
 :: Session -> ByteString -> (Example -> IO a) -> IO a
withExample ses dat = 
  bracket (createExample ses dat) (deleteExample ses)
```
Note: 
- we can add some basic exception handling with brackets
- with these two functions we have enough to be dangerous


# 
```haskell
import Data.ByteString.Char8 (ByteString)
import qualified Policy.Framework.VW.FFI as F

processLines
  ::  ByteString                           
  -> (F.Session -> F.Example -> IO ())             -- 1        
  -> (F.Example -> IO b)              
  -> [ByteString]                              
  -> IO [b]
processLines opts act out dats =
  F.withSession opts                               -- 2
    (\ses -> forM dats                               
      (\bs -> F.withExample ses bs
        (\ex -> act ses ex >> out ex)))            -- 3
```
Note: here's an example of a function i could write w/ my low-level API
- consume an action function, either predict, learn, or some combination
- using withSession to configure vw and generate a session handle ses
- for each example in the dataset i call withExample
- finally using the action to learn/predict and call out on the example pointer
- bracketing twice, once at the session level and once at the example level


# 
```haskell
ghci> dat = "1 |s a^the n^man |t a^un n^homme"
ghci> opts = "--quiet -q st --noconstant"
ghci> act ses ex = F.learn ses ex >> F.predict ses ex
```
```haskell
ghci> out = F.getScalarPrediction 
ghci> :t out
out :: F.Example -> IO Float
```
<!-- .element: class="fragment" -->
```haskell 
ghci> processLines opts act out $ replicate 1 dat
[0.7567993]
```
<!-- .element: class="fragment" -->
```haskell
ghci> processLines opts act out $ replicate 5 dat
[0.7567993,0.9384576,0.9843892,0.9960399,0.99899566]
```
<!-- .element: class="fragment" -->
Note: let's see how this works
- i have a bytestring dat in vw input format (which derives from old libsvm data format). 
- i have a label '1' and two named groups of features
- i have some CL args
- i have an action function that updates an internal vector of weights and makes the updated weights available in an example buffer.
- here's an extractor function, which copies the example buffer out of VW memory and produces a float
- if i train and predict on one copy of the data i get a probability, this is b/c vw has decided that what i wanted to do was train a logistic regression model
- you can see if i train on multiple copies then the probs get progressively closer to 1 as i overfit the data

this is still too low-level though. configuration is implicit and there are no static checks of any kind.
- my input is an untyped list of bytestring
- my options are untyped bytestring
- count the things are happening implicitly: 
- feature engineering (one-hot, quadratic combination),
- class label, s & t are named feature groups, -q st combines the groups quadratically
- the text is interpreted as a the names of boolean features
- the a and n are part-of-speech prefixes, making one-hot encoded categorical variables
- model selection,  logistic classifier is the default, we specify no const term
- model and gradient descent hyper-params


# 
```haskell
import Data.ByteString.Char8 (ByteString)
import qualified Data.ByteString.Char8 as BS8
import qualified Policy.Framework.VW.FFI as F

processLines
  ::  ByteString                           
  -> (F.Session -> F.Example -> IO ())             -- 2    
  -> (F.Example -> IO o)                           -- 3
  -> [ByteString]                              
  -> IO [o]
processLines opts act out dats =
  F.withSession opts                               -- 1
    (\ses -> forM dats                               
      (\bs -> F.withExample ses bs
        (\ex -> act ses ex >> out ex)))
```
Note: 
- using these brackets we end up w/ code written in a cps style
- we also have a common pattern in the type signatures of the arguments
- Example -> IO something 
- we can use this to abstract away some of the cps pattern behind a continuation monad 
- then build some combinators in that monad to provide some compile-time guaruntees.


# 
```haskell
ghci> :t ContT
ContT :: ((e -> m o) -> m o) -> ContT o m e
```
```haskell
ghci> :t ContT @_ @IO @Example
ContT @_ @IO @Example
 :: ((Example -> IO o) -> IO o) -> ContT o IO Example
```
<!-- .element: class="fragment" -->
```haskell
ghci> :t withExample 
withExample
 :: Session -> ByteString -> (Example -> IO o) -> IO o
```
<!-- .element: class="fragment" -->
```haskell
ghci> :t \ses -> ContT . withExample ses
\ses -> ContT . withExample ses
 :: Session -> ByteString -> ContT o IO Example
```
<!-- .element: class="fragment" -->
Note: 
- ContT is a monad transformer that abstracts away continuations
- it says that if you give me a a -> m r, then i'll give you an m r. which if you're familiar w/ the yoneda lemma is a lot like having an a.
- adding some type applications to specialize to our types
- compare this to withExample we just need to lift the last two pieces into the ContT
- second half of withEx is the same as the first half of ContT
- finally it consumes the bytestring and is still polymorphic in the return type r
- going to give us the ability to package bracketed exampl epointer operations with extractor functions and configure them together so that i can check at compile time that the types line w/ what VW is going to expect


# 
```haskell
-- general type for configuring a VW handle
type VW i o e = F.Session -> i -> ContT o IO e
```
```haskell
serialize :: forall o. VW ByteString o F.Example
serialize ses = ContT . F.withExample ses
```
<!-- .element: class="fragment" -->
```haskell
lift ::  
     => (F.Session -> F.Example -> IO ()) 
     -> (F.Example -> IO o) 
     ->  VW F.Example F.Example o
lift act out ses e = liftIO $ act ses e >> out e
```
<!-- .element: class="fragment" -->

Note: 
- here is the basic type i want 
- here is the first restriction note the forall, means that if i have a functions thats just serializing bytestring data into an example pointer, then it must support continuations that produce any output type. 
- i can also take that pattern we used earlier and lift it into a VW type as well


# 
```haskell
import Control.Monad.Trans.Cont
                  
processLines'
  ::  ByteString                           
  -> (F.Session -> F.Example -> IO ())             
  -> (F.Example -> IO b)                           
  -> [ByteString]                              
  ->  IO [b]
processLines' opts act out dats =
  F.withSession opts $ \ses -> 
    evalContT $ do
      exs <- forM dats $ serialize ses 
      forM exs $ lift act out ses
```
Note: 
- if i rewrite processLines serialize and lift the body of the function looks more monadic
- for this function it's not clear how much of an improvement this it, but this isn't the target
- but the actual payoff is that i can build more complex serializers 
- and we'll see in a moment that i can use the type parameters in VW to enforce nice things 


# 
```haskell
import Data.Tuple.Strict (Pair(..))
import Data.Vector.Storable (Vector)

type Example' = T.Pair Example Example            -- 1

serialize' 
 => forall o. VW (NonEmpty Vector ByteString) o Example'
serialize' ses (b :| bs) = do
    e  <- serialize ses b            
    e1 <- lift F.learn return ses e               -- 2
    es <- forM  bs $ serialize ses    
    _  <- forM_ es $ lift F.learn return ses      -- 3
    e2 <- serialize ses ""                        -- 4
    return $ e1 :!: e2 
```

Note: here's the serializer function for the CB ADF use case i mentioned earlier 
- ADF requires non-empty vector of data inputs and a pair of example pointers that all interact in an odd way. 
- operates on the second 'Example' and stores results in the first 'Example'
- A strict pair of 'Example' pointers to be used with ADF contextual bandit models only.
- Parse the first line and hold onto its pointer.
- Parse each remaining line, discarding pointers.
- Parse an empty line, finalising the example.


# 
```haskell
module Policy.Model.VW.Config where

data VWModelConfig i j o e = 
  VWModelConfig { options :: ByteString
                , input   :: F.LabelType
                , output  :: F.PredictType
                , learn   :: VW e () ()            -- 1
                , learn'  :: forall o. VW i o e    -- 2
                , score   :: VW e o o              -- 3
                , score'  :: forall o. VW j o e    -- 4
                }
```
Note: finally we can create a model configuration type
- the first three fields we've already seen
- the last four fields encode all of the constraints we care about.
- my two serializers learn' and score' arent allowed to touch the output, and my learn and score functions must release any pointer memory


# 
```haskell
import qualified Policy.Framework.VW.FFI as F

newtype Finalizer = Finalizer { unFinalizer :: IO () }

instance ModelConfig (VWModelConfig i j o e) where
  type LInput (VWModelConfig i j o e) = i
  type SInput (VWModelConfig i j o e) = j
  type Output (VWModelConfig i j o e) = o
  type Error  (VWModelConfig i j o e) = F.Statistics --!
  type Finalizer (VWModelConfig i j o e) = Finalizer

  createModelHandle = createModelHandle' 
  deleteModelHandle = unFinalizer
```
Note: 
- e refers to the type of the serialized example, all VW models produce the same 
- unlike TF, this instance of createModelHandle takes no arguments. there's no build graph to define
- probably skip the details of creation and go to usage


# 
```haskell
mkFinalizer :: F.Session -> Finalizer
mkFinalizer = Finalizer . F.deleteSession

createModelHandle' ::
       VWModelConfig a p
    -> IO (Handle a p, Finalizer)
createModelHandle' config = do
  ses <- newUnsafeVW opts it ot 
  return (mkHandle ses sl l sp p, mkFinalizer s)
  where opts = options config
        it = itype  config
        ot = otype  config
        l  = learn  config
        sl = learn' config
        p  = score  config
        sp = score' config
```
Note:
- newUnsafeVW just uses the opts to initiate a session and then checks to see that the it and ot match w/ what VW has written into it.
- the finalizer comes directly from the ffi function


# 
```haskell
import Control.Monad ((>=>))

mkHandle :: Session
         -> Serializer i ex
         -> Learner ex
         -> Serializer j ex
         -> Predictor ex o
         -> ModelHandle i j o F.Statistics IO
mkHandle sess sl l sp p = 
  ModelHandle l' p' (F.getStatistics sess) 
  where
    l' = evalContT . (sl ses >=> l ses)
    p' = evalContT . (sp ses >=> p ses)
```
Note: finally mkHandle puts the handle together by evaluating the continuations to construct the learn and score functions, and calling the FFI for the error function. the monad is bound to IO b/c we're using the FFI directly 


