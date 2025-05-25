import gulp from 'gulp';
import shell from 'gulp-shell';

// Build & Serve
gulp.task('build', shell.task('parcel build index.html'));
gulp.task('serve', shell.task('parcel index.html'));

// Testing
gulp.task('test', shell.task('mocha'));
gulp.task('e2e_test', shell.task('npx cypress run'));

// Default => Build & Serve
gulp.task('default', gulp.series('build', 'serve'));