# C: Refinement Types w/ Prisms

Note:


# 

```haskell
module Policy.Types.Numeric where

-- | A real number in the interval [0,1]. 
newtype UnitFloat = UnitFloat { unUnitFloat :: Float }
newtype Probability = 
  Probability { unProbability :: UnitFloat } 
```
```haskell
_UnitFloat :: Prism' Float UnitFloat
_UnitFloat = prism' unUnitFloat perhaps 
  where
    perhaps = bool Nothing <$> (Just . UnitFloat) 
                           <*> inRange 0 1
```
<!-- .element: class="fragment" -->

Note: 
- we use prisms for domain-restricted types like probabilities.
- what is a prism?
- alot of code written in a point-free style, here perhaps just determines whether the 
  input is in the 0 1 range, and return nothing if it isnt


# 

```haskell
inRange :: Ord a => a -> a -> a -> Bool
inRange low high = (&&) <$> (low <=) <*> (<= high)
```

Note: 
- so for example inRange itself is written like so
- like liftA2 from cats or scalaz


# 

```haskell
ghci> :t (<*>) @((->) Float)
(<*>) @((->) Float)
  :: (Float -> a -> b) -> (Float -> a) -> Float -> b
```
<!-- .element: class="fragment" -->
```haskell
ghci> :t bool Nothing <$> (Just . UnitFloat) 
bool Nothing <$> (Just . UnitFloat)
  :: Float -> Bool -> Maybe UnitFloat
```
<!-- .element: class="fragment" -->
```haskell
ghci> :t (<*>) (bool Nothing <$> (Just . UnitFloat))
(<*>) (bool Nothing <$> (Just . UnitFloat))
  :: (Float -> Bool) -> Float -> Maybe UnitFloat
```
<!-- .element: class="fragment" -->
```haskell
ghci> :t bool Nothing <$> (Just . UnitFloat) <*> inRange 0 1
bool Nothing <$> (Just . UnitFloat) <*> inRange 0 1
  :: Float -> Maybe UnitFloat
```
<!-- .element: class="fragment" -->

Note: let's walk through the bool example
- 


# 

```haskell
module Policy.Types.Numeric where

-- | A real number in the interval [-1,1]. 
newtype BiUnitFloat = 
  BiUnitFloat { unBiUnitFloat :: Float }
newtype Reward = Reward { unReward :: BiUnitFloat } 

_BiUnitFloat :: Prism' Float BiUnitFloat
_BiUnitFloat = prism' unBiUnitFloat perhaps
  where 
    perhaps = bool Nothing <$> (Just . BiUnitFloat) 
                           <*> inRange (-1) 1
```

Note: 
- a biunit float is in -1 1.
- we use these for rewards to RL agents. RL algorithms are notoriously sensitve to gradient vector sizes


# 

```haskell
newtype Action = Action { unAction :: Word32 }
```
```haskell
-- | Off-policy logged feedback to an agent. 
data Arp = 
  Arp { _a :: Action, _r :: Reward, _p :: Probability } 
makeLenses ''Arp
```
<!-- .element: class="fragment" -->

Note: 


# 

```haskell
import Control.Lens ((^.), view, to, re)           -- 2

ips :: 
  forall i o. (Get TN.Arp i, Get TN.Action o) 
  => i -> o -> Float                               -- 1
ips input output = bool 0.0 awarded $ act == label ^. a 
  where 
    act = output ^. get
    label = input ^. get
    awarded = award label
    award = (/) <$> view (r . to unReward . re _BiUnitFloat) 
                <*> view (p . to unProbability . re _UnitFloat)
```

Note: what is ips?
- this are where the RLData constraint is used 
- view is how we use a getter, carrot-dot is just in infix version of view
- what is re? vs review?
