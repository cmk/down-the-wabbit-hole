# ML Architecture 

![](ml-arch.png) 
<!-- .element: class="fragment" -->


# Links

https://jaspervdj.be/posts/2018-03-08-handle-pattern.html
https://www.schoolofhaskell.com/user/meiersi/the-service-pattern
https://hackernoon.com/the-has-type-class-pattern-ca12adab70ae
http://www.parsonsmatt.org/2018/03/22/three_layer_haskell_cake.html
https://www.fpcomplete.com/blog/2017/06/readert-design-pattern
https://www.tweag.io/posts/2018-04-25-funflow.html


Note:

Pretty much any sort of Haskell code can be written in this particular way, but that doesn’t mean that you should. This method relies heavily on IO. Whenever you can write things in a pure way, you should attempt to do that and avoid IO. This pattern is only useful when IO is required


