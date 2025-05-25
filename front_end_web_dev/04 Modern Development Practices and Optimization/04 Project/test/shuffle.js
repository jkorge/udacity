import {expect} from 'chai';
import {shuffle} from '../src/shuffle.js';

describe('Shuffle Function', () => {

    it('Should shuffle array indices', () => {

        const cards = [1, 2, 3, 4];
        const shuffled = shuffle(cards);

        // Type check
        expect(shuffled).to.be.an('array');

        // Make sure all the elements are still there
        expect(shuffled.toSorted()).to.deep.equal(cards);

        // Make sure they aren't in the same order
        expect(shuffled).to.not.equal(cards);
    })
})