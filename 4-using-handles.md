# Using Handles

Note:
- this section presents code from our codebase, some info obscured and slightly edited for clarity
- still some idiosyncratic patterns, which i'll try and explain
- ok if the implementation doesn't make much sense, important thing is to focus on the types / constraints


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


# OK 
```haskell
newtype Loss = Loss { unLoss :: Float }

-- | Diagnostic information from learning actions.
data LInfo = LInfo {
      numExamples :: !Word64
    , averageLoss :: !Loss
    } deriving (Eq, Ord, Show)
```
<!-- .element: class="fragment" -->
```haskell
type LLogger c m = M.ModelHandle' c m 
                        -> LInfo 
                        -> M.TrainingInput c 
                        -> m LInfo
```
<!-- .element: class="fragment" -->

Note:  
- talk about sgd
- logger for learning phase of sgd
- learn gets avg monoid?
- LLogger uses the handle on the input and updates the LearnInfo


# 

```haskell
data HInfo = HInfo {
       totalHoldout :: !Word64
     , sumLoss :: !Loss
     } deriving 

type HLogger = HInfo -> IO () 
```

Note: 
-- For holdout only. for RL pipelines we use a single scoring framework, which helps us compare TF and VW models. Pure haskell not configuration dependant
-- for learning only


# 
```haskell
updateLearnInfo ::
  forall c m.                                      -- 1
  (M.ModelConfig c, Get Loss (M.Error c), MonadIO m)
  => Verbosity
  -> LLogger c m 
updateLearnInfo v handle info input = 
  if check v (numExamples info) then t else pure info t
  where
   t = do
     stats <- M.error handle input                 -- 3
     let i = info { averageLoss = stats ^. get }   -- 4
     logLearningInfo i                             -- 5
     pure i

check :: Verbosity -> Word64 -> Bool
check (Verbosity per _) seen = seen `mod` per == 0 -- 2
```

Note: here's an example of an learn logger. note the get constraint.
- forall means that this works for any config type c and any monad m such that
- a check function looks at the number of examples seen so far, then either passes the info object along unchanged or else updates it, according to some verbosity setting
- to update, we call the error function from our handle 
- use the getter from the typeclass to produce a Loss 
- write it to a log somewhere, need the monadIO constraint here


# 

```haskell
module Policy.Learner.Train.RL where

import Control.Monad.Trans.Class (lift)
import Data.Conduit (ConduitT,(.|))                -- 1
import qualified Data.Conduit.Combinators as C
import qualified Policy.Model.Types as M
```
```haskell
type CT i o m r= ConduitT i o m r                  -- 2

learn :: forall c m. (M.ModelConfig c, MonadIO m)
  => M.ModelHandle' c m
  -> LLogger c m
  -> CT (Word64, M.TrainingInput c) LInfo (ExceptT LError m) ()
learn handle logger = 
     C.iterM (lift . M.learn handle . snd)         -- 3
  .| maintainInfo @c (logger handle)               -- 4
```
<!-- .element: class="fragment" -->
```haskell
C.iterM :: Monad m => (a -> m ()) -> CT a a m ()
```
<!-- .element: class="fragment" -->

Note: the type sig requires that this function work for all model configs w/ a monadIO instance
- what is conduit? resource-managed streams, dot-pipe connects them, just like FS2 from scala
- ConduitT input output monad return. making an alias to keep code on screen as much as possible
  here input is (index, data), ouput is info, monad is a transformer stack, no final return value
- take the training input and use the handle to learn on it, lift into ExceptT
- maintainInfo just executes training logger function on the info and passes it along


# 

```haskell
{-# LANGUAGE ConstraintKinds #-}                   -- 1
import qualified Policy.Model.Types as M
import qualified Policy.Types.Numeric as TN        -- 5

type RLModel c = 
  ( M.ModelConfig c, 
    Get (M.ServingInput c) (M.TrainingInput c)     -- 2
  )
```
```haskell
type RLData c = 
  ( M.ModelConfig c, 
    Get TN.Action (M.Output c),                    -- 3
    Get TN.Arp (M.TrainingInput c)                 -- 4
  )
```
<!-- .element: class="fragment" -->

Note: if I want to do holdout though i need constraints
- what is ConstraintKinds?
- this constraint stipulates that we be able to isolate a set of unlabeled features from the. it ensures that we don't accidentally pass labelled data to the agent. we've actually been bitten by this before. 
- this constraint says that the model output must have a selected an action for the agent to take
- an Arp is a special kind of training input for doing off-policy learning with an inverse propensity sampler
- Arps: it's important that training data itself be validated. we have a special set of types for that which i'll discuss in a bit


# 

```haskell
holdout :: forall c m. (RLData c, RLModel c, MonadIO m)
  => M.ModelHandle' c m
  -> HLogger
  -> CT (Word64, M.TrainingInput c) HInfo (ExceptT LError m) ()
holdout handle logger = 
       C.mapM score                                -- 1
    .| holdoutUpdate @c                            -- 3
    .| holdoutReport logger
  where score (idx, dat) = 
    do out <- lift $ M.score handle (dat ^. get)    -- 2
       pure (idx, dat, out)
```
```haskell
C.mapM :: Monad m => (a -> m b) -> CT a b m ()
```
<!-- .element: class="fragment" -->

Note: 
-- score is a conduit that consumes an index-input tuple and outputs a triple of index, input and the RL agent's selected action
-- this is where we use the Get (M.ServingInput c) (M.TrainingInput c) part of the RLModel constraint. we cant run the M.score function directly on in, since it is a training input. 
   the get constraint shouldn't ever appear in model-serving code, since we don't have labels available at serving time.
-- finally, since by design the model itself has no way of determining its hold-out performance (since it never saw a label), we are going to determine holdout loss in a model-agnostic fashion
 score function returns a output that we're going to compare to the input in order to assess performance thus far. we could use the results of this to implement an early stopping criterion for a training job for example. i'll return to this part in a moment.


# 

```haskell
holdoutUpdate :: forall c m. (RLData c, MonadIO m) -- 1
  => CT (Word64, M.TrainingInput c, M.Output c) HInfo m ()
holdoutUpdate = void $ C.scan updateHInfo mempty   -- 2
  where 
    updateHInfo (_, in, out) !acc =
      let delta = ips in out                       -- 3
      in acc `mappend` HInfo 1 (realToFrac delta)  -- 4
```
```haskell
C.scan :: Monad m => (a -> b -> b) -> b -> CT a b m b
```
<!-- .element: class="fragment" -->

```haskell
ips :: forall i o. (Get TN.Arp i, Get TN.Action o)
    => i -> o -> Loss
```
<!-- .element: class="fragment" -->

Note: 
-- We no longer need the RLModel constraint b/c the model has already acted. we still need the RLData constraint for our ips function
-- C.scan?
-- this is the critical part, requiring access to an arp. the definition of ips requires 
   a fair amount of explanation, so i'm going to skip it for now. there are slides on it in appendix C. happy to discuss after the talk though if anyone is interested
-- HInfo is analagous to LInfo, both have monoid instances


# Putting It All Together

```haskell
data TrainingHandle c m = TrainingHandle {
      holdoutStrategy :: HoldoutStrategy
    , holdoutLogger   :: HLogger            
    , learningLogger  :: LLogger c m        
    }
```
<!-- .element: class="fragment" -->

Note: 
-- For holdout only. Pure haskell not configuration dependant
-- for learning only


# 
```haskell
type TInfo = Either HInfo LInfo                    -- 

trainFoldM :: 
  forall c m. (RLData c, RLModel c, MonadIO m)
  => M.ModelHandle' c m
  -> TrainingHandle c m                            -- 
  -> CT (M.TrainingInput c) TInfo (ExceptT LError m) ()
trainFoldM mh th =
       issueId 0
    .| holdoutBranch (holdoutStrategy th)          -- 
    .| holdoutOrLearn                              -- 
    .| gatherInfo
  where
    holdoutOrLearn = getZipConduit $ l <* r        --
    l = leftOnly $ holdout @c mh (holdoutLogger  th)
    r = rightOnly $  learn @c mh (learningLogger th)
    leftOnly  = ZipConduit . C.filter isLeft
    rightOnly = ZipConduit . C.filter isRight
```

Note: doesn't need monadMask b/c it's already inside a mask


# 

```haskell
import Control.Monad.Trans.Control (liftWith)

train :: 
  forall c m. (RLData c, RLModel c, MonadIO m, MonadMask m)
  => c                                                            
  -> TrainingHandle c m                                               
  -> CT () (M.TrainingInput c) (ExceptT LError m) ()
  -> ExceptT LError m ()
train mc lh inputs = void $
  liftWith $ \run -> M.withModelConfig mc $ run . act
  where 
    runFold s s' = C.runConduit (s .| s' .| C.last)
    act :: M.ModelHandle' c m -> ExceptT LearnerError m ()
    act h = do
      minfo <- runFold inputs $ trainFoldM @c h lh
      let tinfo = fromMaybe mempty minfo
      logTrainingInfo tinfo
      pure ()
```

Note: 
- can use the withModelConfig function we defined earlier to bracket the operation
- monadMask required to run withModelConfig
- last :: Monad m => ConduitT a o m (Maybe a)
- runConduit :: Monad m => ConduitT () Void m r -> m r
- liftWith :: (MonadTransControl t, Monad m) => (Run t -> m a) -> t m a 
  lifts everything into 


