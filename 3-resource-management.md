# Creating a TF Handle

Note:


# 
```haskell
module Policy.Model.TF.Simple.Config where

import qualified TensorFlow.Core as TF
                              
data Simple = Simple { unSimple :: TFModelConfig } -- 1

type Float' = TF.TensorData Float                  -- 2

class ModelConfig SimpleTF where
    type ServingInput  Simple = Float'            
    type TrainingInput Simple = (Float', Float')   -- 3
    type Output        Simple = Float             
    type Error         Simple = Float

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
-- TF.build :: Build a -> SessionT m a

newtype SessionT m a = 
  Session (ReaderT SessionState (BuildT m) a)
  deriving (Monad, MonadIO, MonadMask)

type Session = SessionT IO

newtype BuildT m a = BuildT (StateT GraphState m a)

type Build = BuildT Identity
```
Note:


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
-- Build is a State monad over the graph nodes. I'm simply creating the TF graph here. this requires no data dependencies
-- want variable input so i dont' couple my model definition to training details
-- render the output variables representing the dependant variable and error
-- configure my optimization step. standard sgd is fine for something this simple. for more complex models i would probably want to store the opt hyperparameters in the configuration type.
-- ran out of room but the creating the model handle is simply a matter of attaching the rendered variables to the x and y data feeds
          learn = \(x',y') -> TF.runWithFeeds_ [TF.feed x x', TF.feed y y'] learn'
        , score = \x' -> TF.unScalar <$> TF.runWithFeeds [TF.feed x x'] score'
        , error = \(x',y') -> TF.unScalar <$> TF.runWithFeeds [TF.feed x x', TF.feed y y'] error'
-- Theres a TF model config more complex but lets stick to a simpler framework


# Creating a VW Handle

Note:
under the hood both frameworks are wrapping C & C++ code. the main difference is that TF was intended to be used w/ language bindings, wheras VW configurability is limited a (very large) set of configuration flags, essentially an ADT

What is Vowpal Wabbit? - John Langford , Fast gradient descent library, bandits


# 

```
# Test 19: neural network 3-parity with 2 hidden units
vw  -k -c -d train-sets/3parity --hash all --passes 3000 \ 
    -b 16 --nn 2 -l 10 --invariant -f models/0021.model \ 
    --random_seed 19 --quiet --holdout_off
      train-sets/ref/3parity.stderr
```
<!-- .element: class="fragment" -->
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

Note: most relevant for us is the contextual bandit use case. 
- we're reading in a data file cb_test.ldf and training an adf model w/ sgd
- we're also generating predictions and error / stats logs as we go
- the aim of the rest of this talk is to show how we productionize this exact use case


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
-- opaque pointer alias
-- common trick
-- label and predict are large sum types of maybe a dozen options each, depending on the configuration options 
-- so the main things that go into a session are input type, output type, and a string w/ command line options flags, which is unfortunate but VW's internal representation of options is just a bytestring


#   
```haskell
module Policy.Framework.VW.FFI where

learn :: Session -> Example -> IO ()               
predict :: Session -> Example -> IO ()             
getStatistics :: Session -> IO Statistics                 
```
```haskell
newSession :: ByteString -> IO Session             
deleteSession :: Session -> IO ()
cloneSession :: Session -> IO Session

readExample :: Session -> ByteString -> IO Example
finishExample :: Session -> Example -> IO ()
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
withSession :: ByteString -> (Session -> IO a) -> IO a
withSession args = 
  Ex.bracket (newSession args) deleteSession

withExample 
 :: Session -> ByteString -> (Example -> IO a) -> IO a
withExample ses line = 
  Ex.bracket (readExample ses line) (finishExample ses)
```
Note: with these two functions we finally have enough to be dangerous


# 
```haskell
import Data.ByteString.Char8 (ByteString)
import qualified Data.ByteString.Char8 as BS8
import qualified Policy.Framework.VW.FFI as F

processLines
  ::  ByteString                           
  -> (F.Session -> F.Example -> IO ())            
  -> (F.Example -> IO b)
  -> [ByteString]                              
  -> IO [b]
processLines opts act predict dats =
  F.withSession opts                               
  (\ses -> forM dats                               
    (\bs -> F.withExample ses bs
      (\ex -> act ses ex >> predict ex)))
```
Note:
- using withSession describe action
- not great still b/c my options are untyped bytestring
- my input is an untyped list of bytestring
- im handling raw Sessions 


# 
```haskell
ghci> dat = "1 |s w^the w^man |t w^un w^homme"
ghci> opts = "--quiet -q st --noconstant"
ghci> act ses ex = F.learn ses ex *> F.predict ses ex
```
```haskell
ghci> pred = F.getScalarPrediction 
ghci> :t pred
pred :: F.Example -> IO Float
```
<!-- .element: class="fragment" -->
```haskell 
ghci> processLines opts act pred $ replicate 1 dat
[0.85572296]
```
<!-- .element: class="fragment" -->
```haskell
ghci> processLines opts act pred $ replicate 5 dat
[0.85572296,0.9787713,0.9968768,0.9995403,0.99993217]
```
<!-- .element: class="fragment" -->
Note: 
- dat is in vw input format, which derives from old libsvm data format. 
- if i train and predict on one copy of the data i get a probability
- if i train on multiple copies then the probs get progressively closer to 1 as i overfit the data
- count the things are happening implicitly: 
    feature engineering (one-hot, quadratic combination),
      class label, |s & |t are named feature groups, -q st combines the groups quadratically
      the text is interpreted as a the names of boolean features
      the p and w are part-of-speech prefixes, making one-hot encoded categorical variables
 
    model selection, 
       logistic classifier is the default, we specify no const term

    model and gradient descent hyper-params


```haskell
import Data.ByteString.Char8 (ByteString)
import qualified Data.ByteString.Char8 as BS8
import qualified Policy.Framework.VW.FFI as F

processLines
  ::  ByteString                           
  -> (F.Session -> F.Example -> IO ())            
  -> (F.Example -> IO b)
  -> [ByteString]                              
  -> IO [b]
processLines opts act predict dats =
  F.withSession opts                               
  (\ses -> forM dats                               
    (\bs -> F.withExample ses bs
      (\ex -> act ses ex >> predict ex)))
```
Note: notice that processLines is written in a cps style 


# 
```haskell
import Control.Monad.Trans.Cont

processLines'
  ::  ByteString
  -> (F.Session -> F.Example -> IO ())
  -> (F.Example -> IO b)
  -> [ByteString]
  -> IO [b]
processLines' opts act predict dats = evalContT $ do
  ses <- ContT $ F.withSession opts 
  exs <- forM dats (ContT . F.withExample ses)
  liftIO $ forM exs (\ex -> act ses ex >> predict ex)
```
Note: we can use  build a suite of serializers and predictors that use this
- why do i need 2 forMs?


# 
```haskell
ghci> :t ContT
ContT :: ((a -> m r) -> m r) -> ContT r m a
```
```haskell
ghci> :t ContT @_ @IO @F.Example
ContT @_ @IO @F.Example
  :: ((F.Example -> IO r) -> IO r) -> ContT r IO F.Example
```
<!-- .element: class="fragment" -->
```haskell
ghci> :t F.withExample 
F.withExample
  :: F.Session -> ByteString -> (F.Example -> IO r) -> IO r
```
<!-- .element: class="fragment" -->
```haskell
ghci> :t \ses -> ContT . F.withExample ses
\ses -> ContT . F.withExample ses
  :: F.Session -> ByteString -> ContT r IO F.Example
```
<!-- .element: class="fragment" -->
Note:
- 
- finally it consumes the bytestring and is still polymorphic in the return type r


# 
```haskell
module Policy.Model.VW.Internal where

import Control.Monad ((>=>))
import Policy.Framework.VW.FFI (Session, Example)
import qualified Policy.Framework.VW.FFI as F

-- general type for configuring a VW handle
type VW i o e = Session -> i -> ContT o IO e
```
```haskell
-- serialize i to e and yield for consumption by VW
type Serializer i e = forall o. VW i o e  
```
<!-- .element: class="fragment" -->
```haskell
-- consume a raw input for learning only
type Learner i = VW i () ()

-- score a raw input i and produce an output label o
type Predictor i o = VW i o o
```
<!-- .element: class="fragment" -->
Note:
- note the forall, means that it must work w/ any output type
- learner creates no output and yields no resource
- predictor generates 


# 
```haskell
byteStringSerializer :: Serializer ByteString Example
byteStringSerializer ses = ContT . F.withExample ses
```
```haskell
serializeThen 
  :: (Session -> Example -> IO ()) 
  -> Serializer Example Example
serializeThen f ses e = liftIO $ f ses e >> return e
```
<!-- .element: class="fragment" -->
```haskell
learnThen :: (Example -> IO e) -> Serializer Example e
learnThen f ses e = 
  serializeThen F.learn ses e >>= liftIO . f
```
<!-- .element: class="fragment" -->
Note:
-- execute a low-level VW function on an in-memory 'Example'. 
-- combine a learning 'Serializer' with an output function 'f'.


# 
```haskell
pureSerializer :: (i -> j) -> Serializer i j
pureSerializer f = const $ \i -> return (f i)
```
```haskell
withSerializer :: Serializer i j -> VW j o e -> VW i o e
withSerializer s vw ses = (s ses) >=> (vw ses)
```
<!-- .element: class="fragment" -->
```haskell
inputSerializer 
  :: (i -> ByteString) -> Serializer i Example
inputSerializer f = 
  withSerializer (pureSerializer f) byteStringSerializer
```
<!-- .element: class="fragment" -->
Note:
- lift a pure function into a serializer
- precompose w/ a function 


# 
```haskell
import Data.Tuple.Strict (Pair(..))

type Example' = T.Pair Example Example            -- 1

byteStringSerializer' 
  :: Traversable t 
  => Serializer (NonEmpty t ByteString) Example'
byteStringSerializer' ses (x :| xs) = do
    b  <- byteStringSerializer ses x            
    e1 <- learnThen return ses b                  -- 2
    bs <- forM  xs $ byteStringSerializer ses    
    _  <- forM_ bs $ learnThen return ses         -- 3
    e2 <- byteStringSerializer ses ""             -- 4
    return $ e1 :!: e2 
```
Note:
serialize a (ADF-formatted) 'ByteString' into a VW 'Example'' pointer. For ADF use only.
In ADF mode, VW operates on the second 'Example' and stores results in the first 'Example'.

- A strict pair of 'Example' pointers to be used with ADF contextual bandit models only.
- Parse the first line and hold onto its pointer.
- Parse each remaining line, discarding pointers.
- Parse an empty line, finalising the example.


# 
```haskell
module Policy.Model.VW.Config where

data VWModelConfig i j o = 
  forall e. VWModelConfig { options :: ByteString
                          , input   :: F.LabelType
                          , output  :: F.PredictType
                          , learn   :: T.Serializer i e
                          , learn'  :: T.Learner    e
                          , score   :: T.Serializer j e
                          , score'  :: T.Predictor  e o
                          }
```
Note:


# 
```haskell
import qualified Policy.Framework.VW.FFI as F

newtype Finalizer = Finalizer { unFinalizer :: IO () }

instance ModelConfig (VWModelConfig i j o) where
    type TrainingInput (VWModelConfig i j o) = i
    type ServingInput  (VWModelConfig i j o) = j
    type Output        (VWModelConfig i j o) = o
    type Error     (VWModelConfig i j o) = F.Statistics
    type Finalizer (VWModelConfig i j o) = Finalizer

    createModelHandle = createModelHandle' 
    deleteModelHandle = unFinalizer
```
Note: 
- unlike TF, this instance of createModelHandle takes no arguments. there's no build graph to define


# 
```haskell
mkFinalizer :: F.Session -> Finalizer
mkFinalizer = Finalizer . F.deleteSession

createModelHandle' ::
       VWModelConfig a p
    -> IO (Handle a p, Finalizer)
createModelHandle' config = do
  ses <- newUnsafeVW opts it ot 
  return (mkHandle ses l sl p sp, mkFinalizer s)
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
mkHandle :: Session
         -> Serializer i ex
         -> Learner ex
         -> Serializer j ex
         -> Predictor ex o
         -> ModelHandle i j o F.Statistics IO
mkHandle sess l sl p sp = 
  ModelHandle l' p' (F.getStatistics sess) 
  where
    l' = evalContT $ sl sess >=> l sess
    p' = evalContT $ sp sess >=> l sess
```
Note: finally mkHandle puts the handle together by evaluating the continuations to construct the learn and score functions, and calling the FFI for the error function. the monad is bound to IO b/c we're using the FFI directly 


