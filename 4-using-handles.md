# Using Handles

Note:
- this section presents code from our codebase, some info obscured and slightly edited for clarity
- important thing is to focus on the types / constraints


# Has Pattern 

```haskell
import Control.Lens (Getter, Lens', Prism')

-- https://github.com/ekmett/lens/wiki/Types
class HasFoo s where
  foo :: Lens' s Foo   -- (s -> Foo, Foo -> s -> s)

class AsFoo s where
  _Foo :: Prism' s Foo -- (s -> Maybe Foo, Foo -> s)
```
<!-- .element: class="fragment" -->
```haskell
class GetFoo s where
  foo :: Getter s Foo  -- (s -> Foo)
```
<!-- .element: class="fragment" -->
```haskell
class Get a s where get :: Getter s a
```
<!-- .element: class="fragment" -->

Note:
- one pattern that is very useful is to supply an interface 
- essentially Haskell versions of the visitor pattern
- use template haskell to generate Has/As classes
- we also write slightly weaker form of Has* called Get*
- we actually use a version w/ two type params, flipped so that we can partially apply the class on the target type, similar to the versions above.


## Learning Logger
```haskell
import qualified Policy.Model.Types as M

newtype Loss = Loss { unLoss :: Float }

-- | Diagnostic information from learning actions.
data LInfo = LInfo {
    numExamples :: !Word64
  , modelLoss   :: !Loss
  } deriving (Eq, Ord, Show)

instance Monoid LInfo where ...
```
<!-- .element: class="fragment" -->
```haskell
type LLogger c m = 
  M.ModelHandle' c m -> M.LInput c -> LInfo -> m LInfo
```
<!-- .element: class="fragment" -->

Note:  
- generic code for running sgd jobs. it runs this jobs 'online' so to speak, meaning that we'll go back and forth between learning and scoring holdout data rather than doing all the holdout at the end. 
- this approach makes using hadoop streaming and haskell very convenient.
- LInfo keeps track of the number of examples seen and the average trainging error during the learning phase of sgd
- LLogger uses the handle on the input and updates the LearnInfo


# 
```haskell
updateLearnInfo ::
  forall c m.                                      -- 1
  (M.ModelConfig c, Get Loss (M.Error c), MonadIO m)
  => Verbosity
  -> LLogger c m 
updateLearnInfo v handle input info = 
  if check v (numExamples info) then t else pure info
  where
   t = do
     err <- M.error handle input                   -- 2
     let i = info { modelLoss = err ^. get }       -- 3
     logLearningInfo i                             -- 4
     pure i                                        -- 5
```
```haskell
check :: Verbosity -> Word64 -> Bool
check (Verbosity per _) seen = seen `mod` per == 0
```
<!-- .element: class="fragment" -->

Note: here's an example of an learn logger. note the get constraint.
- forall means that this works for any config type c and any monad m such that
  so long as the error type of my modelConfig is able to generate a Loss value. 
- to update, we call the error function from our handle 
- use the getter from the typeclass to produce a Loss. for a TF model where the error was highly configurable this might just be an id function. for VW every instance is going to return a big statistics type, so the get instance here would compute the average loss from the statistics
- write it to a log somewhere, need the monadIO constraint here
- lift the result into the monad provided by the model handle. for vw is IO for tf is session
- a check function looks at the number of examples seen so far, then either passes the info object along unchanged or else updates it, according to some verbosity setting


# 

```haskell
module Policy.Learner.Train.RL where

import Control.Monad.Trans.Class (lift)
import Data.Conduit (ConduitT,(.|))                
import qualified Data.Conduit.Combinators as C
import qualified Policy.Model.Types as M
```
```haskell
type CT i o m r = ConduitT i o m r                 -- 1
```
<!-- .element: class="fragment" -->
```haskell
learn :: forall c m. (M.ModelConfig c, MonadIO m)
  => M.ModelHandle' c m
  -> LLogger c m                                   -- 2
  -> CT (Word64, M.LInput c) LInfo (ExceptT LError m) ()
learn handle logger = 
     C.iterM (lift . M.learn handle . snd)         -- 3
  .| void (C.scanM (logger handle) mempty)         -- 4
```
<!-- .element: class="fragment" -->
```haskell
iterM :: Monad m => (a -> m ()) -> CT a a m ()
scanM :: Monad m => (a -> b -> m b) -> b -> CT a b m b
```
<!-- .element: class="fragment" -->

Note: this is some library code that trains reinforcement learners
- what is conduit? resource-managed streams, dot-pipe connects them like a unix pipe, just like FS2 from scala
- ConduitT input output monad return. making an alias to keep code on screen as much as possible
  here input is (index, data), ouput is info, monad is a transformer stack, no final return value
- take the training input and use the handle to learn on it, want everything in ExceptT so lift into it if the model handle isnt already using it
- iterM keeps executing the handles learn funcing on the learning input and passing the input along
- scanM is like a fold that generates intermediate values inside a monad. void discards the monadic return
- finally note that learn isn't actually being passed c at all! but it's still able to do all of this type inference on it. that's b/c we're going to use a type application when we cann learn. 


## Scoring Logger
```haskell
data SInfo = SInfo {
       numExamples :: !Word64
     , scoringLoss :: !Loss
     } deriving 

instance Monoid SInfo where ...

type SLogger = SInfo -> IO () 
```

Note: 
- SInfo is like LInfo but for tracking holdout statistics. 
- main diff is that we are going to compute holdout loss ourselves rather than use the model handles error function
- two reasons for using a single scoring framework, 
- helps us compare models fairly.
- for some models the scoring input doesn't include a label of anykind so theres no way for it to generate a holdout score. 
- this is for production model configs for example. ensures that we don't accidentally pass labelled data to the agent.


# 

```haskell
{-# LANGUAGE ConstraintKinds #-}                   -- 4
import qualified Policy.Model.Types as M
import qualified Policy.Types.Numeric as TN        

type Holdout c = 
  ( M.ModelConfig c, 
    Get (M.SInput c) (M.LInput c)                  -- 1
  )
```
```haskell
type Ips c = 
  ( M.ModelConfig c, 
    Get TN.Action (M.Output c),                    -- 2
    Get TN.Arp    (M.LInput c)                     -- 3
  )
```
<!-- .element: class="fragment" -->

Note: if I want to do holdout though i need constraints
- recall that we're sending training data through these streams, but i can only pass scoring input to the score function. so we need some way of getting a scoring input from a training input. for our simple TF model this would just mean dropping the first float in the tuple.   for a RL model it is significantly more complex.
- but we still need to do evaluate the holdout score, that's what the Ips constraint allows for. ips stands for inverse propensity scorer, which is a way of doing off-policy learning with an 
- Arp stands for action reward probability. 
- ConstraintKinds is a language extension that allows us to create special kinds that represnt constraints on polymorpic types.


# 

```haskell
holdout :: forall c m. (Ips c, Holdout c, MonadIO m)
  => M.ModelHandle' c m
  -> SLogger
  -> CT (Word64, M.LInput c) SInfo (ExceptT LError m) ()
holdout handle logger = 
       C.mapM score                                -- 1
    .| holdoutUpdate @c                            -- 3
    .| holdoutReport logger
  where score (idx, dat) = 
    do act <- lift $ M.score handle (dat ^. get)   -- 2
       pure (idx, dat, act)
```
```haskell
mapM :: Monad m => (a -> m b) -> CT a b m ()
```
<!-- .element: class="fragment" -->

Note: 
- score is a conduit that consumes an index-input tuple and outputs a triple of index, input and the RL agent's selected action
- this is where we use the Holdout constraint. we cant run the M.score function directly on in, since it is a training input. 
- the Holdout constraint shouldn't ever appear in model-serving code, since we don't have labels available at serving time.
- finally, since by design the model itself has no way of determining its hold-out performance (since it never saw a label), we are going to determine holdout loss in a model-agnostic fashion
 score function returns an action that we're going to compare to the input in order to assess performance thus far. 


# 

```haskell
holdoutUpdate :: forall c m. (Ips c, MonadIO m)    -- 1
  => CT (Word64, M.LInput c, M.Output c) SInfo m ()
holdoutUpdate = void $ C.scan updateSInfo mempty   -- 2
  where 
    updateSInfo (_, dat, act) !acc =
      let delta = ips dat act                      -- 3
      in acc `mappend` SInfo 1 (realToFrac delta)  -- 4
```
```haskell
scan :: Monad m => (a -> b -> b) -> b -> CT a b m b
```
<!-- .element: class="fragment" -->

```haskell
ips :: forall i o. (Get TN.Arp i, Get TN.Action o)
    => i -> o -> Loss
```
<!-- .element: class="fragment" -->

Note: 
- We no longer need the Holdout constraint b/c the model has already acted. we still need the Ips constraint for our ips function
- C.scan is similar to scanM from before, but a pure version
- this is the critical part, requiring access to an arp. the definition of ips requires 
   a fair amount of explanation, so i'm going to skip it for now. there are slides on it in appendix C. happy to discuss after the talk though if anyone is interested
- SInfo is analagous to LInfo, both have monoid instances
- we could use the results of this to implement an early stopping criterion for a training job for example. i'll return to this part in a moment.


# Putting It All Together

```haskell
data TrainingLogger c m = TrainingLogger {
      trainingStrategy :: Strategy
    , scoringLogger    :: SLogger            
    , learningLogger   :: LLogger c m        
    }
```
<!-- .element: class="fragment" -->

Note: 
-- For holdout only. Pure haskell not configuration dependant
-- for learning only


#  
```haskell
type TInfo = (SInfo, LInfo)                        

trainFoldM :: 
  forall c m. (Ips c, Holdout c, MonadIO m)
  => M.ModelHandle' c m
  -> TrainingLogger c m                            -- 1
  -> CT (M.LInput c) TInfo (ExceptT LError m) ()
trainFoldM mh th =
       issueId 0
    .| holdoutBranch (trainingStrategy th)         -- 2
    .| holdoutOrLearn                              -- 3
    .| gatherInfo
  where
    holdoutOrLearn = getZipConduit $ l <* r        -- 4
    l = leftOnly $ holdout @c mh (scoringLogger  th)
    r = rightOnly $  learn @c mh (learningLogger th)
    leftOnly  = ZipConduit . C.filter isLeft
    rightOnly = ZipConduit . C.filter isRight
```

Note: 
- use trainingLogger to determine when to learn / score, and log each
- holdout or learn branches the stream and calls the correct function
- does this with an applicative instance
- note that trainFoldM has all these reqs on c, but c isn't actually present at the value level. same was true of holdout and learn in fact


# 
```haskell
type TInfo = (SInfo, LInfo)

gatherInfo :: forall m. MonadIO m
           => CT (Either SInfo LInfo) TInfo m ()
gatherInfo = void $ C.scan gather mempty
  where
    gather (Left s')  !(s,l) = (s `mappend` s', l)
    gather (Right l') !(s,l) = (s, l `mappend` l')
```
Note:


# 

```haskell
import Control.Monad.Trans.Control (liftWith)      -- 3

train :: 
  forall c m. (Ips c, Holdout c, MonadIO m, MonadMask m)
  => c                                                            
  -> TrainingLogger c m                                               
  -> CT () (M.LInput c) (ExceptT LError m) ()      -- 1
  -> ExceptT LError m ()
train mc lh inputs = void $
  liftWith $ \r -> M.withModelConfig mc $ r . act  -- 2
  where 
    runFold s s' = C.runConduit (s .| s' .| C.last)
    act mh = do                                    -- 4
      minfo <- runFold inputs $ trainFoldM @c mh lh 
      let tinfo = fromMaybe mempty minfo
      logTrainingInfo tinfo
      pure ()                                      
```
```haskell
runConduit :: Monad m => CT () Void m r -> m r
last :: Monad m => CT i o m (Maybe i)
```
<!-- .element: class="fragment" -->

Note: 
- takes a stream of data inputs from somewhere (hadoop shuffle step)
- i told you way back at slide 10 or so that we would use withModelConfig before the end. this is the end. 
- recall that monadMask was required to run withModelConfig. liftWith lifts the entire computation into a masked state. this takes care of async exception handling.
- runConduit executes the entire stream, returning the r 
- we'd kept r empty the entire time. last is going to grab the accumulated logs
- we write the logs out (or empties if they dont exist) and exit!


