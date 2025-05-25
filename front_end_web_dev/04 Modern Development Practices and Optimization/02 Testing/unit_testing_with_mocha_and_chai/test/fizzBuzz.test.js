import {expect} from 'chai';
import {fizzBuzz} from '../src/fizzBuzz.js';

describe('FizzBuzz Function', () => {

    it('Should return "FizzBuzz" if the input is divisble by 3 and 5', () => {
        expect(fizzBuzz(15)).to.equal('FizzBuzz');
        expect(fizzBuzz(30)).to.equal('FizzBuzz');
    });

    it('Should return "Fizz" if the input is divisble by 3', () => {
        expect(fizzBuzz(6)).to.equal('Fizz');
        expect(fizzBuzz(27)).to.equal('Fizz');
    });

    it('Should return "Buzz" if the input is divisble by 5', () => {
        expect(fizzBuzz(25)).to.equal('Buzz');
        expect(fizzBuzz(55)).to.equal('Buzz');
    });

    it('Should return false if the input is not a number', () => {
        expect(fizzBuzz('not a number')).to.be.false;
        expect(fizzBuzz([])).to.be.false;
        expect(fizzBuzz(true)).to.be.false;
    });

})