const gulp = require('gulp');
const shell = require('gulp-shell');

gulp.task('unit test', shell.task('mocha'));
