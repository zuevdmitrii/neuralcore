!function(n){var e={};function t(r){if(e[r])return e[r].exports;var o=e[r]={i:r,l:!1,exports:{}};return n[r].call(o.exports,o,o.exports,t),o.l=!0,o.exports}t.m=n,t.c=e,t.d=function(n,e,r){t.o(n,e)||Object.defineProperty(n,e,{enumerable:!0,get:r})},t.r=function(n){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(n,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(n,"__esModule",{value:!0})},t.t=function(n,e){if(1&e&&(n=t(n)),8&e)return n;if(4&e&&"object"==typeof n&&n&&n.__esModule)return n;var r=Object.create(null);if(t.r(r),Object.defineProperty(r,"default",{enumerable:!0,value:n}),2&e&&"string"!=typeof n)for(var o in n)t.d(r,o,function(e){return n[e]}.bind(null,o));return r},t.n=function(n){var e=n&&n.__esModule?function(){return n.default}:function(){return n};return t.d(e,"a",e),e},t.o=function(n,e){return Object.prototype.hasOwnProperty.call(n,e)},t.p="",t(t.s=0)}([function(n,e,t){"use strict";t.r(e);var r=function(n){return 1/(1+Math.exp(-n))};var o=function(n){var e=this;this.out=NaN,this.in=NaN,this.derivative=0,this.calculate=function(){e.out=e.activationFn(e.in)},this.activationFn=n.activationFn},a=function(n){this.neurons=[];for(var e=0;e<n.countOfNeurons;e++)this.neurons.push(new o({activationFn:n.isInput?function(n){return n}:r}))},i=function(){var n=this;this.weights={},this.add=function(e,t,r,o){n.weights[e]||(n.weights[e]={}),n.weights[e][t]||(n.weights[e][t]={}),n.weights[e][t][r+"-"+o]={w:Math.random(),wPrev:0,deltaW:0,deltaWPrev:0}},this.get=function(e,t,r,o){return n.weights[e][t][r+"-"+o]}},u=new function(){var n=this;this.layers=[],this.weights=new i,this.Speed=.7,this.Alpha=.01,this.ErrorRate=.01,this.addLayer=function(e){var t=n.layers[n.layers.length-1],r=n.layers.length-1,o=new a({countOfNeurons:e,isInput:!t});n.layers.push(o);var i=n.layers.length-1;if(t)for(var u=t.neurons.length,s=0;s<u;s++)for(var l=0;l<e;l++)n.weights.add(r,i,s,l)},this.calculate=function(e){e.forEach((function(e,t){n.layers[0].neurons[t].in=e,n.layers[0].neurons[t].calculate()}));for(var t=function(e){var t=n.layers[e],r=n.layers[e-1];t.neurons.forEach((function(t,o){var a=r.neurons.reduce((function(t,r,a){return t+n.weights.get(e-1,e,a,o).w*r.out}),0);t.in=a,t.calculate()}))},r=1;r<n.layers.length;r++)t(r);return n.layers[n.layers.length-1].neurons.map((function(n){return n.out}))},this.train=function(e,t){e.forEach((function(e,t){n.layers[0].neurons[t].in=e,n.layers[0].neurons[t].calculate()}));for(var r=function(e){var t=n.layers[e],r=n.layers[e-1];t.neurons.forEach((function(t,o){var a=r.neurons.reduce((function(t,r,a){return t+n.weights.get(e-1,e,a,o).w*r.out}),0);t.in=a,t.calculate()}))},o=1;o<n.layers.length;o++)r(o);var a=n.layers[n.layers.length-1];n.Error=a.neurons.reduce((function(n,e,r){return e.derivative=e.out*(1-e.out)*(t[r]-e.out),n+Math.pow(t[r]-e.out,2)}),0)/a.neurons.length;var i=function(e){var t=n.layers[e],r=n.layers[e+1];t.neurons.forEach((function(t,o){t.derivative=t.out*(1-t.out)*r.neurons.reduce((function(r,a,i){var u=n.weights.get(e,e+1,o,i),s=u.w;return u.deltaW=n.Alpha*u.deltaW+(1-n.Alpha)*n.Speed*(a.derivative*t.out),u.w=u.w+u.deltaW,r+a.derivative*s}),0)}))};for(o=n.layers.length-2;o>=0;o--)i(o)},this.Error=0};u.addLayer(3),u.addLayer(4),u.addLayer(4);var s=[[0,0,0],[0,1,0],[1,0,0],[1,1,0],[0,0,1],[0,1,1],[1,0,1],[1,1,1]],l=[[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]],c=function(){var n=0;s.forEach((function(e,t){u.train(e,l[t]),n+=u.Error})),console.log(n/s.length)},f=function(){console.log("--------------------------------------------"),s.forEach((function(n,e){var t=u.calculate(n);console.log("Data: ",n,"Answer: ",t)})),console.log("--------------------------------------------"),console.log("")};window.startTrain=function(){f();for(var n=0;n<1e3;n++)c();f()},window.NetworkInst=u,console.log(u)}]);
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8vd2VicGFjay9ib290c3RyYXAiLCJ3ZWJwYWNrOi8vLy4vc3JjL2ZuLnRzIiwid2VicGFjazovLy8uL3NyYy9pbmRleC50cyJdLCJuYW1lcyI6WyJpbnN0YWxsZWRNb2R1bGVzIiwiX193ZWJwYWNrX3JlcXVpcmVfXyIsIm1vZHVsZUlkIiwiZXhwb3J0cyIsIm1vZHVsZSIsImkiLCJsIiwibW9kdWxlcyIsImNhbGwiLCJtIiwiYyIsImQiLCJuYW1lIiwiZ2V0dGVyIiwibyIsIk9iamVjdCIsImRlZmluZVByb3BlcnR5IiwiZW51bWVyYWJsZSIsImdldCIsInIiLCJTeW1ib2wiLCJ0b1N0cmluZ1RhZyIsInZhbHVlIiwidCIsIm1vZGUiLCJfX2VzTW9kdWxlIiwibnMiLCJjcmVhdGUiLCJrZXkiLCJiaW5kIiwibiIsIm9iamVjdCIsInByb3BlcnR5IiwicHJvdG90eXBlIiwiaGFzT3duUHJvcGVydHkiLCJwIiwicyIsIlNpZ21vaWQiLCJ4IiwiTWF0aCIsImV4cCIsInByb3BzIiwib3V0IiwiTmFOIiwiaW4iLCJkZXJpdmF0aXZlIiwiY2FsY3VsYXRlIiwiYWN0aXZhdGlvbkZuIiwidGhpcyIsIm5ldXJvbnMiLCJjb3VudE9mTmV1cm9ucyIsInB1c2giLCJOZXVyb24iLCJpc0lucHV0Iiwid2VpZ2h0cyIsImFkZCIsImxheWVyRnJvbSIsImxheWVyVG8iLCJqIiwidyIsInJhbmRvbSIsIndQcmV2IiwiZGVsdGFXIiwiZGVsdGFXUHJldiIsIk5ldHdvcmtJbnN0IiwibGF5ZXJzIiwiR3JhcGgiLCJTcGVlZCIsIkFscGhhIiwiRXJyb3JSYXRlIiwiYWRkTGF5ZXIiLCJsYXN0TGF5ZXIiLCJsZW5ndGgiLCJuZXdMYXllciIsImlDb3VudCIsImlucHV0cyIsImZvckVhY2giLCJpbmRleCIsImN1cnJlbnRMYXllciIsInByZXZMYXllciIsIm5ldXJvbkRlc3QiLCJpbmRleERlc3QiLCJpblZhbHVlIiwicmVkdWNlIiwibmV1cm9uUHJldiIsImluZGV4UHJldiIsIm1hcCIsIm5ldXJvbiIsInRyYWluIiwiYW5zd2VycyIsImxhc3RMYXllcnMiLCJFcnJvciIsImluZGV4QW5zIiwicG93IiwibmV4dExheWVyIiwibGVmdEluZGV4IiwicmlnaHROZXVyb24iLCJyaWdodEluZGV4Iiwid09iaiIsIndDdXJyZW50IiwidHJhaW5EYXRhIiwiYW5zd2VyIiwiZXBvaGUiLCJFcnIiLCJkYXRhIiwiY29uc29sZSIsImxvZyIsImNhbGMiLCJ3aW5kb3ciLCJzdGFydFRyYWluIiwiayJdLCJtYXBwaW5ncyI6ImFBQ0UsSUFBSUEsRUFBbUIsR0FHdkIsU0FBU0MsRUFBb0JDLEdBRzVCLEdBQUdGLEVBQWlCRSxHQUNuQixPQUFPRixFQUFpQkUsR0FBVUMsUUFHbkMsSUFBSUMsRUFBU0osRUFBaUJFLEdBQVksQ0FDekNHLEVBQUdILEVBQ0hJLEdBQUcsRUFDSEgsUUFBUyxJQVVWLE9BTkFJLEVBQVFMLEdBQVVNLEtBQUtKLEVBQU9ELFFBQVNDLEVBQVFBLEVBQU9ELFFBQVNGLEdBRy9ERyxFQUFPRSxHQUFJLEVBR0pGLEVBQU9ELFFBS2ZGLEVBQW9CUSxFQUFJRixFQUd4Qk4sRUFBb0JTLEVBQUlWLEVBR3hCQyxFQUFvQlUsRUFBSSxTQUFTUixFQUFTUyxFQUFNQyxHQUMzQ1osRUFBb0JhLEVBQUVYLEVBQVNTLElBQ2xDRyxPQUFPQyxlQUFlYixFQUFTUyxFQUFNLENBQUVLLFlBQVksRUFBTUMsSUFBS0wsS0FLaEVaLEVBQW9Ca0IsRUFBSSxTQUFTaEIsR0FDWCxvQkFBWGlCLFFBQTBCQSxPQUFPQyxhQUMxQ04sT0FBT0MsZUFBZWIsRUFBU2lCLE9BQU9DLFlBQWEsQ0FBRUMsTUFBTyxXQUU3RFAsT0FBT0MsZUFBZWIsRUFBUyxhQUFjLENBQUVtQixPQUFPLEtBUXZEckIsRUFBb0JzQixFQUFJLFNBQVNELEVBQU9FLEdBRXZDLEdBRFUsRUFBUEEsSUFBVUYsRUFBUXJCLEVBQW9CcUIsSUFDL0IsRUFBUEUsRUFBVSxPQUFPRixFQUNwQixHQUFXLEVBQVBFLEdBQThC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LEVBQ3JCeUMsUUFBUUMsSUFBSTFDIiwiZmlsZSI6ImxpYi5qcyIsInNvdXJjZXNDb250ZW50IjpbIiBcdC8vIFRoZSBtb2R1bGUgY2FjaGVcbiBcdHZhciBpbnN0YWxsZWRNb2R1bGVzID0ge307XG5cbiBcdC8vIFRoZSByZXF1aXJlIGZ1bmN0aW9uXG4gXHRmdW5jdGlvbiBfX3dlYnBhY2tfcmVxdWlyZV9fKG1vZHVsZUlkKSB7XG5cbiBcdFx0Ly8gQ2hlY2sgaWYgbW9kdWxlIGlzIGluIGNhY2hlXG4gXHRcdGlmKGluc3RhbGxlZE1vZHVsZXNbbW9kdWxlSWRdKSB7XG4gXHRcdFx0cmV0dXJuIGluc3RhbGxlZE1vZHVsZXNbbW9kdWxlSWRdLmV4cG9ydHM7XG4gXHRcdH1cbiBcdFx0Ly8gQ3JlYXRlIGEgbmV3IG1vZHVsZSAoYW5kIHB1dCBpdCBpbnRvIHRoZSBjYWNoZSlcbiBcdFx0dmFyIG1vZHVsZSA9IGluc3RhbGxlZE1vZHVsZXNbbW9kdWxlSWRdID0ge1xuIFx0XHRcdGk6IG1vZHVsZUlkLFxuIFx0XHRcdGw6IGZhbHNlLFxuIFx0XHRcdGV4cG9ydHM6IHt9XG4gXHRcdH07XG5cbiBcdFx0Ly8gRXhlY3V0ZSB0aGUgbW9kdWxlIGZ1bmN0aW9uXG4gXHRcdG1vZHVsZXNbbW9kdWxlSWRdLmNhbGwobW9kdWxlLmV4cG9ydHMsIG1vZHVsZSwgbW9kdWxlLmV4cG9ydHMsIF9fd2VicGFja19yZXF1aXJlX18pO1xuXG4gXHRcdC8vIEZsYWcgdGhlIG1vZHVsZSBhcyBsb2FkZWRcbiBcdFx0bW9kdWxlLmwgPSB0cnVlO1xuXG4gXHRcdC8vIFJldHVybiB0aGUgZXhwb3J0cyBvZiB0aGUgbW9kdWxlXG4gXHRcdHJldHVybiBtb2R1bGUuZXhwb3J0cztcbiBcdH1cblxuXG4gXHQvLyBleHBvc2UgdGhlIG1vZHVsZXMgb2JqZWN0IChfX3dlYnBhY2tfbW9kdWxlc19fKVxuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy5tID0gbW9kdWxlcztcblxuIFx0Ly8gZXhwb3NlIHRoZSBtb2R1bGUgY2FjaGVcbiBcdF9fd2VicGFja19yZXF1aXJlX18uYyA9IGluc3RhbGxlZE1vZHVsZXM7XG5cbiBcdC8vIGRlZmluZSBnZXR0ZXIgZnVuY3Rpb24gZm9yIGhhcm1vbnkgZXhwb3J0c1xuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy5kID0gZnVuY3Rpb24oZXhwb3J0cywgbmFtZSwgZ2V0dGVyKSB7XG4gXHRcdGlmKCFfX3dlYnBhY2tfcmVxdWlyZV9fLm8oZXhwb3J0cywgbmFtZSkpIHtcbiBcdFx0XHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgbmFtZSwgeyBlbnVtZXJhYmxlOiB0cnVlLCBnZXQ6IGdldHRlciB9KTtcbiBcdFx0fVxuIFx0fTtcblxuIFx0Ly8gZGVmaW5lIF9fZXNNb2R1bGUgb24gZXhwb3J0c1xuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy5yID0gZnVuY3Rpb24oZXhwb3J0cykge1xuIFx0XHRpZih0eXBlb2YgU3ltYm9sICE9PSAndW5kZWZpbmVkJyAmJiBTeW1ib2wudG9TdHJpbmdUYWcpIHtcbiBcdFx0XHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgU3ltYm9sLnRvU3RyaW5nVGFnLCB7IHZhbHVlOiAnTW9kdWxlJyB9KTtcbiBcdFx0fVxuIFx0XHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgJ19fZXNNb2R1bGUnLCB7IHZhbHVlOiB0cnVlIH0pO1xuIFx0fTtcblxuIFx0Ly8gY3JlYXRlIGEgZmFrZSBuYW1lc3BhY2Ugb2JqZWN0XG4gXHQvLyBtb2RlICYgMTogdmFsdWUgaXMgYSBtb2R1bGUgaWQsIHJlcXVpcmUgaXRcbiBcdC8vIG1vZGUgJiAyOiBtZXJnZSBhbGwgcHJvcGVydGllcyBvZiB2YWx1ZSBpbnRvIHRoZSBuc1xuIFx0Ly8gbW9kZSAmIDQ6IHJldHVybiB2YWx1ZSB3aGVuIGFscmVhZHkgbnMgb2JqZWN0XG4gXHQvLyBtb2RlICYgOHwxOiBiZWhhdmUgbGlrZSByZXF1aXJlXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLnQgPSBmdW5jdGlvbih2YWx1ZSwgbW9kZSkge1xuIFx0XHRpZihtb2RlICYgMSkgdmFsdWUgPSBfX3dlYnBhY2tfcmVxdWlyZV9fKHZhbHVlKTtcbiBcdFx0aWYobW9kZSAmIDgpIHJldHVybiB2YWx1ZTtcbiBcdFx0aWYoKG1vZGUgJiA0KSAmJiB0eXBlb2YgdmFsdWUgPT09ICdvYmplY3QnICYmIHZhbHVlICYmIHZhbHVlLl9fZXNNb2R1bGUpIHJldHVybiB2YWx1ZTtcbiBcdFx0dmFyIG5zID0gT2JqZWN0LmNyZWF0ZShudWxsKTtcbiBcdFx0X193ZWJwYWNrX3JlcXVpcmVfXy5yKG5zKTtcbiBcdFx0T2JqZWN0LmRlZmluZVByb3BlcnR5KG5zLCAnZGVmYXVsdCcsIHsgZW51bWVyYWJsZTogdHJ1ZSwgdmFsdWU6IHZhbHVlIH0pO1xuIFx0XHRpZihtb2RlICYgMiAmJiB0eXBlb2YgdmFsdWUgIT0gJ3N0cmluZycpIGZvcih2YXIga2V5IGluIHZhbHVlKSBfX3dlYnBhY2tfcmVxdWlyZV9fLmQobnMsIGtleSwgZnVuY3Rpb24oa2V5KSB7IHJldHVybiB2YWx1ZVtrZXldOyB9LmJpbmQobnVsbCwga2V5KSk7XG4gXHRcdHJldHVybiBucztcbiBcdH07XG5cbiBcdC8vIGdldERlZmF1bHRFeHBvcnQgZnVuY3Rpb24gZm9yIGNvbXBhdGliaWxpdHkgd2l0aCBub24taGFybW9ueSBtb2R1bGVzXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLm4gPSBmdW5jdGlvbihtb2R1bGUpIHtcbiBcdFx0dmFyIGdldHRlciA9IG1vZHVsZSAmJiBtb2R1bGUuX19lc01vZHVsZSA/XG4gXHRcdFx0ZnVuY3Rpb24gZ2V0RGVmYXVsdCgpIHsgcmV0dXJuIG1vZHVsZVsnZGVmYXVsdCddOyB9IDpcbiBcdFx0XHRmdW5jdGlvbiBnZXRNb2R1bGVFeHBvcnRzKCkgeyByZXR1cm4gbW9kdWxlOyB9O1xuIFx0XHRfX3dlYnBhY2tfcmVxdWlyZV9fLmQoZ2V0dGVyLCAnYScsIGdldHRlcik7XG4gXHRcdHJldHVybiBnZXR0ZXI7XG4gXHR9O1xuXG4gXHQvLyBPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGxcbiBcdF9fd2VicGFja19yZXF1aXJlX18ubyA9IGZ1bmN0aW9uKG9iamVjdCwgcHJvcGVydHkpIHsgcmV0dXJuIE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChvYmplY3QsIHByb3BlcnR5KTsgfTtcblxuIFx0Ly8gX193ZWJwYWNrX3B1YmxpY19wYXRoX19cbiBcdF9fd2VicGFja19yZXF1aXJlX18ucCA9IFwiXCI7XG5cblxuIFx0Ly8gTG9hZCBlbnRyeSBtb2R1bGUgYW5kIHJldHVybiBleHBvcnRzXG4gXHRyZXR1cm4gX193ZWJwYWNrX3JlcXVpcmVfXyhfX3dlYnBhY2tfcmVxdWlyZV9fLnMgPSAwKTtcbiIsImV4cG9ydCBjb25zdCBTaWdtb2lkID0gKHg6IG51bWJlcikgPT4ge1xyXG4gIHJldHVybiAxIC8gKCAxICsgTWF0aC5leHAoLXgpKVxyXG59XHJcblxyXG5leHBvcnQgY29uc3QgR1RhbiA9ICh4OiBudW1iZXIpID0+IHtcclxuICByZXR1cm4gKE1hdGguZXhwKDIgKiB4KSAtIDEpIC8gKE1hdGguZXhwKDIqeCkgKyAxKVxyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gc2lnbW9pZCh4Om51bWJlciwgQTogbnVtYmVyLCAgZGVyaXZhdGl2ZTpib29sZWFuKSB7ICAgICBcclxuICBsZXQgZnggPSAxIC8gKDEgKyBNYXRoLmV4cCgteCAqIDIgKiBBKSk7ICAgICBcclxuICBpZiAoZGVyaXZhdGl2ZSkgICAgICAgICBcclxuICAgICByZXR1cm4gZnggKiAoMSAtIGZ4KTsgICAgIFxyXG4gIHJldHVybiBmeDsgXHJcbn0iLCJpbXBvcnQgeyBTaWdtb2lkLCBHVGFuIH0gZnJvbSBcIi4vZm5cIjtcclxuXHJcbmludGVyZmFjZSBJTmV1cm9uIHtcclxuICBpbjogbnVtYmVyO1xyXG4gIG91dDogbnVtYmVyO1xyXG4gIGFjdGl2YXRpb25GbjogKHg6IG51bWJlcikgPT4gbnVtYmVyO1xyXG4gIGNhbGN1bGF0ZTogKCkgPT4gdm9pZDtcclxuICBkZXJpdmF0aXZlOiBudW1iZXI7XHJcbn1cclxuXHJcbmludGVyZmFjZSBJTGF5ZXIge1xyXG4gIG5ldXJvbnM6IElOZXVyb25bXTtcclxufVxyXG5cclxuaW50ZXJmYWNlIElMYXllclByb3BzIHtcclxuICBjb3VudE9mTmV1cm9uczogbnVtYmVyO1xyXG4gIGlzSW5wdXQ6IGJvb2xlYW47XHJcbn1cclxuXHJcbmludGVyZmFjZSBJTmV0d29yayB7XHJcbiAgbGF5ZXJzOiBJTGF5ZXJbXTtcclxuICB3ZWlnaHRzOiBJR3JhcGg7XHJcbn1cclxuXHJcbmludGVyZmFjZSBJV2VpZ2h0IHtcclxuICB3OiBudW1iZXI7XHJcbiAgd1ByZXY6IG51bWJlcjtcclxuICBkZWx0YVc6IG51bWJlcjtcclxuICBkZWx0YVdQcmV2OiBudW1iZXI7XHJcbn1cclxuXHJcbmludGVyZmFjZSBJR3JhcGgge1xyXG4gIGFkZDogKGxheWVyRnJvbTogbnVtYmVyLCBsYXllclRvOiBudW1iZXIsIGk6IG51bWJlciwgajogbnVtYmVyKSA9PiB2b2lkO1xyXG4gIGdldDogKGxheWVyRnJvbTogbnVtYmVyLCBsYXllclRvOiBudW1iZXIsIGk6IG51bWJlciwgajogbnVtYmVyKSA9PiBJV2VpZ2h0O1xyXG59XHJcblxyXG5pbnRlcmZhY2UgSU5ldXJvblByb3BzIHtcclxuICBhY3RpdmF0aW9uRm46ICh4OiBudW1iZXIpID0+IG51bWJlcjtcclxufVxyXG5cclxuY2xhc3MgTmV1cm9uIGltcGxlbWVudHMgSU5ldXJvbiB7XHJcbiAgcHVibGljIG91dCA9IE5hTjtcclxuICBwdWJsaWMgaW4gPSBOYU47XHJcbiAgcHVibGljIGRlcml2YXRpdmUgPSAwO1xyXG4gIHB1YmxpYyBhY3RpdmF0aW9uRm47XHJcblxyXG4gIHB1YmxpYyBjYWxjdWxhdGUgPSAoKSA9PiB7XHJcbiAgICB0aGlzLm91dCA9IHRoaXMuYWN0aXZhdGlvbkZuKHRoaXMuaW4pO1xyXG4gIH07XHJcblxyXG4gIGNvbnN0cnVjdG9yKHByb3BzOiBJTmV1cm9uUHJvcHMpIHtcclxuICAgIHRoaXMuYWN0aXZhdGlvbkZuID0gcHJvcHMuYWN0aXZhdGlvbkZuO1xyXG4gIH1cclxufVxyXG5cclxuY2xhc3MgTGF5ZXIgaW1wbGVtZW50cyBJTGF5ZXIge1xyXG4gIHB1YmxpYyBuZXVyb25zOiBJTmV1cm9uW10gPSBbXTtcclxuICBjb25zdHJ1Y3Rvcihwcm9wczogSUxheWVyUHJvcHMpIHtcclxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgcHJvcHMuY291bnRPZk5ldXJvbnM7IGkrKykge1xyXG4gICAgICB0aGlzLm5ldXJvbnMucHVzaChcclxuICAgICAgICBuZXcgTmV1cm9uKHtcclxuICAgICAgICAgIGFjdGl2YXRpb25GbjogcHJvcHMuaXNJbnB1dCA/ICh4KSA9PiB4IDogU2lnbW9pZCwgLy9HVGFuLFxyXG4gICAgICAgIH0pXHJcbiAgICAgICk7XHJcbiAgICB9XHJcbiAgfVxyXG59XHJcblxyXG5jbGFzcyBHcmFwaCBpbXBsZW1lbnRzIElHcmFwaCB7XHJcbiAgcHJpdmF0ZSB3ZWlnaHRzOiB7XHJcbiAgICBbcHJvcG5hbWU6IG51bWJlcl06IHtcclxuICAgICAgW3Byb3BuYW1lOiBudW1iZXJdOiB7XHJcbiAgICAgICAgW3Byb3BuYW1lOiBzdHJpbmddOiBJV2VpZ2h0O1xyXG4gICAgICB9O1xyXG4gICAgfTtcclxuICB9ID0ge307XHJcbiAgcHVibGljIGFkZCA9IChsYXllckZyb206IG51bWJlciwgbGF5ZXJUbzogbnVtYmVyLCBpOiBudW1iZXIsIGo6IG51bWJlcikgPT4ge1xyXG4gICAgaWYgKCF0aGlzLndlaWdodHNbbGF5ZXJGcm9tXSkge1xyXG4gICAgICB0aGlzLndlaWdodHNbbGF5ZXJGcm9tXSA9IHt9O1xyXG4gICAgfVxyXG4gICAgaWYgKCF0aGlzLndlaWdodHNbbGF5ZXJGcm9tXVtsYXllclRvXSkge1xyXG4gICAgICB0aGlzLndlaWdodHNbbGF5ZXJGcm9tXVtsYXllclRvXSA9IHt9O1xyXG4gICAgfVxyXG4gICAgdGhpcy53ZWlnaHRzW2xheWVyRnJvbV1bbGF5ZXJUb11bYCR7aX0tJHtqfWBdID0ge1xyXG4gICAgICB3OiBNYXRoLnJhbmRvbSgpLFxyXG4gICAgICB3UHJldjogMCxcclxuICAgICAgZGVsdGFXOiAwLFxyXG4gICAgICBkZWx0YVdQcmV2OiAwLFxyXG4gICAgfTtcclxuICB9O1xyXG5cclxuICBwdWJsaWMgZ2V0ID0gKGxheWVyRnJvbTogbnVtYmVyLCBsYXllclRvOiBudW1iZXIsIGk6IG51bWJlciwgajogbnVtYmVyKSA9PiB7XHJcbiAgICByZXR1cm4gdGhpcy53ZWlnaHRzW2xheWVyRnJvbV1bbGF5ZXJUb11bYCR7aX0tJHtqfWBdO1xyXG4gIH07XHJcbn1cclxuXHJcbmNsYXNzIE5ldHdvcmsgaW1wbGVtZW50cyBJTmV0d29yayB7XHJcbiAgcHVibGljIGxheWVyczogSUxheWVyW10gPSBbXTtcclxuICBwdWJsaWMgd2VpZ2h0cyA9IG5ldyBHcmFwaCgpO1xyXG4gIHB1YmxpYyBTcGVlZDogbnVtYmVyID0gMC43O1xyXG4gIHB1YmxpYyBBbHBoYTogbnVtYmVyID0gMC4wMTtcclxuICBwdWJsaWMgRXJyb3JSYXRlOiBudW1iZXIgPSAwLjAxXHJcblxyXG4gIHB1YmxpYyBhZGRMYXllciA9IChjb3VudE9mTmV1cm9uczogbnVtYmVyKSA9PiB7XHJcbiAgICBjb25zdCBsYXN0TGF5ZXIgPSB0aGlzLmxheWVyc1t0aGlzLmxheWVycy5sZW5ndGggLSAxXTtcclxuICAgIGNvbnN0IGxheWVyRnJvbSA9IHRoaXMubGF5ZXJzLmxlbmd0aCAtIDE7XHJcbiAgICBjb25zdCBuZXdMYXllciA9IG5ldyBMYXllcih7IGNvdW50T2ZOZXVyb25zLCBpc0lucHV0OiAhbGFzdExheWVyIH0pO1xyXG4gICAgdGhpcy5sYXllcnMucHVzaChuZXdMYXllcik7XHJcbiAgICBjb25zdCBsYXllclRvID0gdGhpcy5sYXllcnMubGVuZ3RoIC0gMTtcclxuICAgIGlmIChsYXN0TGF5ZXIpIHtcclxuICAgICAgY29uc3QgaUNvdW50ID0gbGFzdExheWVyLm5ldXJvbnMubGVuZ3RoO1xyXG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IGlDb3VudDsgaSsrKSB7XHJcbiAgICAgICAgZm9yIChsZXQgaiA9IDA7IGogPCBjb3VudE9mTmV1cm9uczsgaisrKSB7XHJcbiAgICAgICAgICB0aGlzLndlaWdodHMuYWRkKGxheWVyRnJvbSwgbGF5ZXJUbywgaSwgaik7XHJcbiAgICAgICAgfVxyXG4gICAgICB9XHJcbiAgICB9XHJcbiAgfVxyXG5cclxuICBwdWJsaWMgY2FsY3VsYXRlID0gKGlucHV0czogbnVtYmVyW10pOm51bWJlcltdID0+IHtcclxuICAgIGlucHV0cy5mb3JFYWNoKCh2YWx1ZSwgaW5kZXgpID0+IHtcclxuICAgICAgdGhpcy5sYXllcnNbMF0ubmV1cm9uc1tpbmRleF0uaW4gPSB2YWx1ZTtcclxuICAgICAgdGhpcy5sYXllcnNbMF0ubmV1cm9uc1tpbmRleF0uY2FsY3VsYXRlKCk7XHJcbiAgICB9KTtcclxuXHJcbiAgICBmb3IgKGxldCBpID0gMTsgaSA8IHRoaXMubGF5ZXJzLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgIGNvbnN0IGN1cnJlbnRMYXllciA9IHRoaXMubGF5ZXJzW2ldO1xyXG4gICAgICBjb25zdCBwcmV2TGF5ZXIgPSB0aGlzLmxheWVyc1tpIC0gMV07XHJcbiAgICAgIGN1cnJlbnRMYXllci5uZXVyb25zLmZvckVhY2goKG5ldXJvbkRlc3QsIGluZGV4RGVzdCkgPT4ge1xyXG4gICAgICAgIGNvbnN0IGluVmFsdWUgPSBwcmV2TGF5ZXIubmV1cm9ucy5yZWR1Y2UoXHJcbiAgICAgICAgICAodmFsdWUsIG5ldXJvblByZXYsIGluZGV4UHJldikgPT4ge1xyXG4gICAgICAgICAgICByZXR1cm4gKFxyXG4gICAgICAgICAgICAgIHZhbHVlICtcclxuICAgICAgICAgICAgICB0aGlzLndlaWdodHMuZ2V0KGkgLSAxLCBpLCBpbmRleFByZXYsIGluZGV4RGVzdCkudyAqXHJcbiAgICAgICAgICAgICAgICBuZXVyb25QcmV2Lm91dFxyXG4gICAgICAgICAgICApO1xyXG4gICAgICAgICAgfSxcclxuICAgICAgICAgIDBcclxuICAgICAgICApO1xyXG4gICAgICAgIG5ldXJvbkRlc3QuaW4gPSBpblZhbHVlO1xyXG4gICAgICAgIG5ldXJvbkRlc3QuY2FsY3VsYXRlKCk7XHJcbiAgICAgIH0pO1xyXG4gICAgfVxyXG5cclxuICAgIGNvbnN0IGxhc3RMYXllcnMgPSB0aGlzLmxheWVyc1t0aGlzLmxheWVycy5sZW5ndGggLSAxXTtcclxuICAgIHJldHVybiBsYXN0TGF5ZXJzLm5ldXJvbnMubWFwKG5ldXJvbiA9PiBuZXVyb24ub3V0KVxyXG4gIH1cclxuXHJcbiAgcHVibGljIHRyYWluID0gKGlucHV0czogbnVtYmVyW10sIGFuc3dlcnM6IG51bWJlcltdKSA9PiB7XHJcbiAgICBpbnB1dHMuZm9yRWFjaCgodmFsdWUsIGluZGV4KSA9PiB7XHJcbiAgICAgIHRoaXMubGF5ZXJzWzBdLm5ldXJvbnNbaW5kZXhdLmluID0gdmFsdWU7XHJcbiAgICAgIHRoaXMubGF5ZXJzWzBdLm5ldXJvbnNbaW5kZXhdLmNhbGN1bGF0ZSgpO1xyXG4gICAgfSk7XHJcblxyXG4gICAgZm9yIChsZXQgaSA9IDE7IGkgPCB0aGlzLmxheWVycy5sZW5ndGg7IGkrKykge1xyXG4gICAgICBjb25zdCBjdXJyZW50TGF5ZXIgPSB0aGlzLmxheWVyc1tpXTtcclxuICAgICAgY29uc3QgcHJldkxheWVyID0gdGhpcy5sYXllcnNbaSAtIDFdO1xyXG4gICAgICBjdXJyZW50TGF5ZXIubmV1cm9ucy5mb3JFYWNoKChuZXVyb25EZXN0LCBpbmRleERlc3QpID0+IHtcclxuICAgICAgICBjb25zdCBpblZhbHVlID0gcHJldkxheWVyLm5ldXJvbnMucmVkdWNlKFxyXG4gICAgICAgICAgKHZhbHVlLCBuZXVyb25QcmV2LCBpbmRleFByZXYpID0+IHtcclxuICAgICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgICB2YWx1ZSArXHJcbiAgICAgICAgICAgICAgdGhpcy53ZWlnaHRzLmdldChpIC0gMSwgaSwgaW5kZXhQcmV2LCBpbmRleERlc3QpLncgKlxyXG4gICAgICAgICAgICAgICAgbmV1cm9uUHJldi5vdXRcclxuICAgICAgICAgICAgKTtcclxuICAgICAgICAgIH0sXHJcbiAgICAgICAgICAwXHJcbiAgICAgICAgKTtcclxuICAgICAgICBuZXVyb25EZXN0LmluID0gaW5WYWx1ZTtcclxuICAgICAgICBuZXVyb25EZXN0LmNhbGN1bGF0ZSgpO1xyXG4gICAgICB9KTtcclxuICAgIH1cclxuXHJcbiAgICBjb25zdCBsYXN0TGF5ZXJzID0gdGhpcy5sYXllcnNbdGhpcy5sYXllcnMubGVuZ3RoIC0gMV07XHJcbiAgICB0aGlzLkVycm9yID1cclxuICAgICAgbGFzdExheWVycy5uZXVyb25zLnJlZHVjZSgodmFsdWUsIG5ldXJvbiwgaW5kZXhBbnMpID0+IHtcclxuICAgICAgICBuZXVyb24uZGVyaXZhdGl2ZSA9XHJcbiAgICAgICAgICBuZXVyb24ub3V0ICogKDEgLSBuZXVyb24ub3V0KSAqIChhbnN3ZXJzW2luZGV4QW5zXSAtIG5ldXJvbi5vdXQpO1xyXG4gICAgICAgICAgXHJcbiAgICAgICAgcmV0dXJuIHZhbHVlICsgTWF0aC5wb3coYW5zd2Vyc1tpbmRleEFuc10gLSBuZXVyb24ub3V0LCAyKTtcclxuICAgICAgfSwgMCkgLyBsYXN0TGF5ZXJzLm5ldXJvbnMubGVuZ3RoO1xyXG5cclxuICAgIC8vaWYgKHRoaXMuRXJyb3IgPCB0aGlzLkVycm9yUmF0ZSkge1xyXG4gICAgLy8gIHJldHVyblxyXG4gICAgLy99XHJcbiAgICBmb3IgKGxldCBpID0gdGhpcy5sYXllcnMubGVuZ3RoIC0gMjsgaSA+PSAwOyBpLS0pIHtcclxuICAgICAgY29uc3QgY3VycmVudExheWVyID0gdGhpcy5sYXllcnNbaV07XHJcbiAgICAgIGNvbnN0IG5leHRMYXllciA9IHRoaXMubGF5ZXJzW2kgKyAxXTtcclxuICAgICAgY3VycmVudExheWVyLm5ldXJvbnMuZm9yRWFjaCgobmV1cm9uLCBsZWZ0SW5kZXgpID0+IHtcclxuICAgICAgICBuZXVyb24uZGVyaXZhdGl2ZSA9XHJcbiAgICAgICAgICBuZXVyb24ub3V0ICpcclxuICAgICAgICAgICgxIC0gbmV1cm9uLm91dCkgKlxyXG4gICAgICAgICAgbmV4dExheWVyLm5ldXJvbnMucmVkdWNlKCh2YWx1ZSwgcmlnaHROZXVyb24sIHJpZ2h0SW5kZXgpID0+IHtcclxuICAgICAgICAgICAgY29uc3Qgd09iaiA9IHRoaXMud2VpZ2h0cy5nZXQoaSwgaSArIDEsIGxlZnRJbmRleCwgcmlnaHRJbmRleClcclxuICAgICAgICAgICAgY29uc3Qgd0N1cnJlbnQgPSB3T2JqLndcclxuXHJcbiAgICAgICAgICAgIHdPYmouZGVsdGFXID0gdGhpcy5BbHBoYSAqIHdPYmouZGVsdGFXICsgKDEgLSB0aGlzLkFscGhhKSAqIHRoaXMuU3BlZWQgKiAocmlnaHROZXVyb24uZGVyaXZhdGl2ZSAqIG5ldXJvbi5vdXQpXHJcbiAgICAgICAgICAgIHdPYmoudyA9IHdPYmoudyArIHdPYmouZGVsdGFXXHJcblxyXG4gICAgICAgICAgICByZXR1cm4gKFxyXG4gICAgICAgICAgICAgIHZhbHVlICtcclxuICAgICAgICAgICAgICByaWdodE5ldXJvbi5kZXJpdmF0aXZlICpcclxuICAgICAgICAgICAgICAgIHdDdXJyZW50XHJcbiAgICAgICAgICAgICk7XHJcbiAgICAgICAgICB9LCAwKTtcclxuICAgICAgfSk7XHJcblxyXG4gICAgfVxyXG4gIH07XHJcblxyXG4gIHB1YmxpYyBFcnJvcjogbnVtYmVyID0gMDtcclxufVxyXG5cclxuY29uc3QgTmV0d29ya0luc3QgPSBuZXcgTmV0d29yaygpO1xyXG5OZXR3b3JrSW5zdC5hZGRMYXllcigzKTtcclxuTmV0d29ya0luc3QuYWRkTGF5ZXIoNCk7XHJcbk5ldHdvcmtJbnN0LmFkZExheWVyKDQpO1xyXG5cclxuY29uc3QgdHJhaW5EYXRhID0gW1xyXG4gWzAsIDAsIDBdLFxyXG4gWzAsIDEsIDBdLFxyXG4gWzEsIDAsIDBdLFxyXG4gWzEsIDEsIDBdLFxyXG4gWzAsIDAsIDFdLFxyXG4gWzAsIDEsIDFdLFxyXG4gWzEsIDAsIDFdLFxyXG4gWzEsIDEsIDFdLFxyXG5dO1xyXG5cclxuY29uc3QgYW5zd2VyID0gW1xyXG4gWzEsIDAsIDAsIDBdLFxyXG4gWzAsIDEsIDAsIDBdLFxyXG4gWzAsIDEsIDAsIDBdLFxyXG4gWzAsIDAsIDEsIDBdLFxyXG4gWzAsIDEsIDAsIDBdLFxyXG4gWzAsIDAsIDEsIDBdLFxyXG4gWzAsIDAsIDEsIDBdLFxyXG4gWzAsIDAsIDAsIDFdLFxyXG5dO1xyXG5cclxuY29uc3QgZXBvaGUgPSAoKSA9PiB7XHJcbiAgbGV0IEVyciA9IDBcclxuICB0cmFpbkRhdGEuZm9yRWFjaCgoZGF0YSwgaW5kZXgpID0+IHtcclxuICAgIE5ldHdvcmtJbnN0LnRyYWluKGRhdGEsIGFuc3dlcltpbmRleF0pO1xyXG4gICAgRXJyICs9IE5ldHdvcmtJbnN0LkVycm9yXHJcbiAgfSlcclxuICBjb25zb2xlLmxvZyhFcnIgLyB0cmFpbkRhdGEubGVuZ3RoKVxyXG59XHJcblxyXG5jb25zdCBjYWxjID0gKCkgPT4ge1xyXG4gIGNvbnNvbGUubG9nKCctLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLScpXHJcbiAgdHJhaW5EYXRhLmZvckVhY2goKGRhdGEsIGluZGV4KSA9PiB7XHJcbiAgICBjb25zdCBhbnN3ZXIgPSBOZXR3b3JrSW5zdC5jYWxjdWxhdGUoZGF0YSlcclxuICAgIGNvbnNvbGUubG9nKCdEYXRhOiAnLCBkYXRhLCAnQW5zd2VyOiAnLCBhbnN3ZXIpXHJcbiAgfSlcclxuICBcclxuICBjb25zb2xlLmxvZygnLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0nKVxyXG4gIGNvbnNvbGUubG9nKCcnKVxyXG59XHJcblxyXG5cclxuLy9AdHMtaWdub3JlXHJcbndpbmRvdy5zdGFydFRyYWluID0gKCkgPT4ge1xyXG4gIGNhbGMoKVxyXG4gIGZvciAobGV0IGs9MDsgazwxMDAwOyBrKyspIHtcclxuICAgIGVwb2hlKClcclxuICB9XHJcbiAgY2FsYygpXHJcbn1cclxuXHJcbi8vQHRzLWlnbm9yZVxyXG53aW5kb3cuTmV0d29ya0luc3QgPSBOZXR3b3JrSW5zdCBcclxuY29uc29sZS5sb2coTmV0d29ya0luc3QpO1xyXG4iXSwic291cmNlUm9vdCI6IiJ9