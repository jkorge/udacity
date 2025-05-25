const path = require("path");

// TODO: Complete this Webpack configuration
// 1. Set `module.exports` to an object literal.
// 2. Add a `mode` key and set its value to `"production"`.
// 3. Add an `entry` key with the value pointing to the path of `index.js`.
// 4. Add an `output` key with the value set to an object literal:
//    - Add a `path` key and set it to `path.resolve(__dirname, "dist")`.
//      - `path.resolve` creates an absolute path by resolving segments into a full path.
//      - `__dirname` provides the absolute path of the current directory.
//      - `"dist"` is the folder we want to point to within that directory.
//    - Add a `filename` key and set it to a JavaScript file name like `main.js` or `bundle.js`.


module.exports = {
    mode: 'production',
    entry: './src/index.js',
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'main.js'
    }
}