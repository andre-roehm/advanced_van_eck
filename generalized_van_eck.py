import os, glob
from array import array

from math import *
from random import *
import numpy as np
import pylab as plt
import sys

#take a numpy sequence called seq and check whether it is consistent with the advanced van eck property
def check_advanced_van_eck_property(seq, verbal=True):
    #get the length of the sequence
    len = np.size(seq)
    #check all the elements of the sequence
    for i in np.arange(len):
        #get the delay
        delay = seq[i]
        if delay == 0:
            #a 0 indicates that the next element (i+1) will be the last time this element occurs. Check that none of the following ones are equal to it
            for j in np.arange(i+2, len):
                if seq[j] == seq[i+1]:
                    if verbal:
                        print("The following element violates the 0-property: position (0-indexed) " + str(i+1) + " is equal to " + str(j) + " with value " + str(seq[i+1]))
                    return False
        #if all the following values are -1, the sequence is over, because this is used as a wildcard
        elif( np.max(seq[i+1:]) == -1):
            return True
        #check if the delayed element is in the sequence (only if the delay is larger than 0)
        if (i + 1 + delay < len) and (delay > 0):
            #the delay then indicates that the following element should be equal to the element at position i+1+delay
            #if the following element violates the property, return false
            if seq[i+1] != seq[i+1+delay]:
                if verbal:
                    print("The following element violates the property: " + str(seq[i+1]) + " != " + str(seq[i+delay+1]) + " with delay " + str(delay) + " at position " + str(i+1) + " as part of the subsequence " + str(seq[i:i+delay+2]))
                return False
        #if the following element appears before the index given by the delay, return false
        for j in np.arange(i+2, min(i+1+delay, len)):
            if seq[j] == seq[i+1]:
                print("The following element violates the property: " + str(seq[j]) + " appears at positiong " + str(j) + " after " + str(i+1) + " but before " + str( i + 1 + delay) + " with delay " + str(delay) + " as part of the subsequence " + str(seq[i:i+delay+2]))
                return False

    #if all the elements of the sequence satisfy the property, return true
    return True


#make a test sequence
test_sequence_1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

test_sequence_2 = np.array([0,0,1,2,2,4,2,4])

test_sequence_3 = np.array([0,0,0])

#print(check_advanced_van_eck_property(test_sequence_1))
#print(check_advanced_van_eck_property(test_sequence_2))
#print(check_advanced_van_eck_property(test_sequence_3))


##
##   Old, first attempts 
##

# #add a new element to the sequence and speculate
# def advanced_van_eck_speculate(seq, start_val = 0, max_spec=10):
#     #speculate for all possible values until one is found that satisfies the property or until all have been tried
#     if start_val > max_spec:
#         return False, seq
#     #we don't have to attempt to fill in values that are larger than the largest value that appears in the sequence + 1 (because that would be the first "new independent" value we can try - and if neither a new, nor an old value is valid, we don't have to try other new values)
    
#     temp_max_spec = np.amax(seq)
#     temp_max_spec = temp_max_spec + 1

#     #if the temp_max_spec is larger than the max_spec, set it to the max_spec
#     if temp_max_spec > max_spec :
#         temp_max_spec = max_spec

#     if temp_max_spec <= start_val:
#         temp_max_spec = start_val + 1

#     for i in (np.arange(start_val, temp_max_spec + 1)):
#         speculative_seq = np.append(seq, i)
#         if check_advanced_van_eck_property(speculative_seq, False):
#             return True, speculative_seq
#     #no speculation was successful
#     return False, seq


# #speculate for n elements without backtracking
# def primitive_speculate_for_n(n, seq):
#     for i in np.arange(n):
#         status, seq = advanced_van_eck_speculate(seq)
#         if(status == False):
#             print("Speculation failed at position " + str(i))
#             return seq
#     return seq

# #print(primitive_speculate_for_n(10, np.array([0,0])))


# def speculate_with_backtracking_loop(start_seq, n, max_spec=10):
   
#     #main loop

#     i = np.size(start_seq)
#     seq = start_seq

#     step_counter = 0

#     while i < n + 1:
#         #check if we are in a backtrack
#         attempt_start_val = 0

#         if np.size(seq) == i + 1:
#             #if we are in a backtrack, get the sequence from the memory
#             #get the last element of the last row of the sequence memory
#             attempt_start_val = seq[-1] + 1

#             #delete the last row of the sequence memory
#             seq = np.delete(seq, -1)
            
#             # if attempt_start_val > max_spec:
#             #     return False, seq

#         status, seq = advanced_van_eck_speculate(seq, start_val=attempt_start_val, max_spec=max_spec)
#         if(status == False):
#             i = i - 1
#             if i < 0:
#                 return False, seq
#         else:
#             i = i + 1
#         step_counter = step_counter + 1

#         if step_counter%1000 == 0:
#             print("Step " + str(step_counter) + " reached, sequence length: " + str(np.size(seq)))
#             print(seq)


#     return True, seq

# #DOESNT WORK
# # def nested_advanced_van_eck_calc(start_seq, n, max_spec):
    
# #     limited_max_spec = 10

# #     seq = start_seq

# #     while limited_max_spec < max_spec:
# #         status, seq = speculate_with_backtracking_loop(seq, n, limited_max_spec)

# #         if status == False:
# #             limited_max_spec = limited_max_spec + 10
# #             print("Increasing max_spec to " + str(limited_max_spec))
# #             print(seq)
# #         else:
# #             return True, seq
    
# #     #reached the max limit
# #     return False, seq


# #very direct way to calculate, very slow, scales poorly wih max_spec
# def advanced_van_eck(start_seq, n, max_spec):
#     status, seq = speculate_with_backtracking_loop(start_seq, n, max_spec)
#     if status == False:
#         print("Speculation failed at length " + str(np.size(seq)))
#         print(seq)
#         return seq
#     else:
#         print("Speculation succeeded")
#         print(seq)
#         return seq



# #advanced_van_eck(np.array([0,0]), 100, 50)

# #print(nested_advanced_van_eck_calc(np.array([0,0]), 10, 20))


##
##  New Approach based on bit-mask elimination
##

## The main idea is to treat it like a Sudoku and eliminate possibilities
## 

#bit-mask based approach

#define a class for the masked_van_eck with various submethods
class masked_van_eck:
    def __init__(self, sequence_max_length, sequence_max_val, seed_sequence):
        self.sequence_max_length = sequence_max_length          #saves the maximum length of the sequence that can be reached in forward search
        self.sequence_max_val = sequence_max_val                #saves the maximum value of the sequence that can be used in forward search
        
        #make the initial sequence vector, entries with -1 are meant to be "unknown"-placeholders
        self.seq = np.ones(sequence_max_length, dtype=int) * -1             #create the main object - the sequence that is being forward continued, set them all to -1 (placeholder for "unknown")
        self.seq[0:np.size(seed_sequence)] = seed_sequence                  #set the seed sequence

        #set up a history of the sequence, needed so that we may be able to revert after a failed speculation
        self.seq_history = np.zeros((1, self.sequence_max_length), dtype=int)
        self.seq_history[0,:] = self.seq

        #generate the bit mask matrix
        self.mask = np.ones((self.sequence_max_length,self.sequence_max_val), dtype=int)
        #the mask is a matrix with the same numberr of rows as the max_length of the sequence, and a number of columns equal to the number of possible guesses. Each entry is either 0 or 1. mask[i,j] = 0 indicates that position i cannot have value j (for some reason, methods to narrow down the search space are used to set values to 0)

        #the mask also needs a history
        self.mask_history = np.zeros((1, self.sequence_max_length, self.sequence_max_val), dtype=int)

        self.finished_steps = 0

        self.guess_position_history = np.zeros((1), dtype=int)
        self.guess_position_history[0] = np.size(seed_sequence) - 1 #this assume that the seed sequence does not contain any gaps. If it does, this needs to be changed

        self.logical_update_position = 1 #this assume that the seed sequence does not contain any gaps. If it does, this needs to be changed

        #calculate the first logical deductions for the mask from the seed sequence
        self.mask_update_forbidden_because_zeros()
        # self.mask_update_forbidden_because_sandwiched_left_moving()  #this function probably does nothing here

        self.mask_history[0,:,:] = self.mask



    #print the sequence without trailing zeroes
    def print_seq_without_trailing_empty(self):
        #find the last element not equal to -1 (our placeholder to indicate an "empty slot") and print the sequence until then
        last_element = np.where(self.seq != -1)[0][-1]
        print(self.seq[0:last_element + 1])



    #functions that update the bit matrix based on the sequence
    def mask_update_forbidden_because_zeros(self):

        #if any entry of the sequence is 0, this means that the value following it is not allowed to ever appear again
        for i in np.arange(np.size(self.seq) - 1):
            if self.seq[i] == 0:
                self.mask[i+1:,self.seq[i+1] ] = 0   #the mask of all positions greater than i+1 are not allowed to be the same as seq[i+1]


    def mask_update_forbidden_because_sandwhiched_individual(self):
        #try all the positions that are currently -1 in our sequence (that is our placeholder for unknown)
        for i in np.where(self.seq == -1)[0]:
            #check all non--1 elements that occur after the current position and see if the same entry of sequence has occured before the current position. 
            #For example for the sequence [1, 2, -1, 2, 3] (where position i here has the entry -1) we want to check if the entry 2 has occured before position i and after position i. A 2 would be forbidden, whereas a 1 or 3 would be fine in this example (because they only appear on one side)
            for j in np.where(self.seq[i+1:] != -1)[0]:
                for k in np.arange(i):
                    if self.seq[k] == self.seq[i+j+1]:
                        self.mask[i, self.seq[k]] = 0
                        break

    # #this function is not needed for the current approach
    # def mask_update_forbidden_because_sandwiched_left_moving(self):
    #     #check for the first element of the sequence that is still -1 (unknown)
    #     first_unknown_element = np.where(self.seq == -1)[0][0]


    #     #check all non--1 elements that occur after the current position 
    #     for j in np.where(self.seq[first_unknown_element+1:] != -1)[0]:
    #         #as long as we assume that the initial seed sequence did not contain any place holders (program needs to be changed otherwise), this position for sure is after the seed sequence.
    #         #for these values, we now can be sure that they were obtained from "speculation" and logical deductions arising from it. The only value that can appear after a "gap" (the empty position indicated by -1), must have been put there by being logically induced through a delay from earlier in the sequence. Thus it must be a re-occurence.
    #         #forbid all the entries in the mask of this value to the left of here
    #         for k in np.arange(first_unknown_element + j + 1):
    #             self.mask[k, self.seq[first_unknown_element + j + 1]] = 0  #this step is setting a lot of mask values to 0 for entries that are already filled, so not very efficient


    #a function that takes a current sequence and a mask and information on the last updated position that was changed by a logical update, and calculating resulting mask by eliminating forbidden positions
    def mask_update_resulting_from_logical_step(self, position):

        if np.size(np.where(self.seq == -1)[0]) == 0:
            return 1 #indicates that we have reached the end of the sequence

        first_unknown_element = np.where(self.seq == -1)[0][0]

        if position < first_unknown_element:
            # print(np.where(self.seq == -1)[0])
            # print("Position: " + str(position) + " is smaller than first unknown element: " + str(first_unknown_element))
            return 1 #no mask update necessary
        
        value_entered = self.seq[position]

        #loop through all unknown positions from the position provided to the last position where the same value appears
        for j in np.where(self.seq[first_unknown_element:position] == -1)[0]:
            self.mask[first_unknown_element + j, value_entered] = 0 #forbid the value entered at these positions, for example  14 4 -1 -1 


    #a function that takes a current sequence and a mask and information on the last updated position, and calculating the resulting sequence and mask
    def logical_one_step_update(self):

        if(self.logical_update_position >= np.where(self.seq != -1)[0][-1] + 1):
            return 5 #indicates that we have reached the end of the known part of the sequence and that there is no point continuing

        if(self.seq[self.logical_update_position] == -1):
            return 3 #indicates that we have a position with no info. If we had a correlation mask matrix, this is where we would update it.
        

        #new value to be added to the sequence
        delay_to_be_used = self.seq[self.logical_update_position - 1]
        if delay_to_be_used == -1:
            return 6 #indicates that we are skipping this position because the position before it is still empty.


        if self.logical_update_position + delay_to_be_used < np.size(self.seq):
            #check if the position is free
            if self.seq[self.logical_update_position + delay_to_be_used] == -1:
                #insert the smallest value possible that is allowed by the mask at position last_updated_position + delay_to_be_used
        
                #check if we are allowed to just copy over the position
                if (self.mask[self.logical_update_position + delay_to_be_used, self.seq[self.logical_update_position]] == 1):

                    self.seq[self.logical_update_position + delay_to_be_used] = self.seq[self.logical_update_position]

                    #update the mask by considering new impossible positions
                    self.mask_update_resulting_from_logical_step(self.logical_update_position + delay_to_be_used)

                    return 1 #this indicates that there might be more changes needed but that the logical step update was successful
                    
                else:
                    
                    # print("Mask-Contradiction at position: ", self.logical_update_position, " taking the value ", self.seq[self.logical_update_position], " and trying to put it to ", self.logical_update_position + delay_to_be_used, " seems forbidden, because the masks is ", self.mask[self.logical_update_position + delay_to_be_used, :])
                    
                    return 0 #False, we have a contradiction (we are not allowed to put this value here despite the target position being empty)


            #otherwise we are finding it in an existing position
            else:
                if(self.seq[self.logical_update_position] == self.seq[self.logical_update_position + delay_to_be_used]):
                    return 2 #we rediscovered a previous fitting value, logically consistent
                else:
                    #we have a contradiction because the position was already filled with a different number
                    # print("Collision-Contradiction at position: ", self.logical_update_position, " taking the value ", self.seq[self.logical_update_position], " to position ", self.logical_update_position + delay_to_be_used, " but there already found ", self.seq[self.logical_update_position + delay_to_be_used])
                    return 0 
        else:
            return 4 #we hit the end of the series, we may want to stop


    #a function that loops over the logical one step updates until we have no more logical changes
    def logical_update_until_convergence(self, active_slot):
        #keep track of the number of iterations
        keep_iterating = True
        self.logical_update_position = active_slot

        # print("Guess position history: " , self.guess_position_history)

        if(self.finished_steps == 0):
            self.logical_update_position = 1

        while keep_iterating and (self.logical_update_position < np.size(self.seq)):

            #update the sequence
            logical_update_result = self.logical_one_step_update()
            if self.finished_steps == 2:
                # print("Logical update result: ", logical_update_result, " at step " , self.logical_update_position) 
                # print("Sequence history : ", self.seq_history[self.finished_steps,:])
                pass

            #check if we need to keep iterating
            if logical_update_result == 0:
                # print("LOGICAL CONTRADICTION")
                return 0 #indicates that we have a contradiction
            
            elif logical_update_result == 4:
                keep_iterating = False
                return 2 #indicates that we have reached the end of the sequence

            elif logical_update_result == 5:
                keep_iterating = False
                return 1
            #These clauses are implicit (partially because we are not using a correlation matrix)
            # elif logical_update_result == 3:
            #     keep_iterating = True
            # elif logical_update_result == 2:
            #     keep_iterating = True
            # elif logical_update_result == 1:
            #     keep_iterating = True

            self.logical_update_position += 1
            # print(self.seq)


    #a function that uses the current state, and goes back in the history one step, resets and correctly updates the mask
    def revert_a_step(self):

        if(self.finished_steps == 0):
            return False
        

        #revert the sequence
        # print ("Reverting to step: ", self.finished_steps - 1)
        # print (self.seq_history)
        # print (self.seq)

        # print(self.mask)
        # print(self.mask_history)




        #make sure we don't repeat the same guess
        self.mask_history[self.finished_steps - 1, self.guess_position_history[self.finished_steps], self.seq[self.guess_position_history[self.finished_steps]] ] = 0

        # print("Last time we tried: ", self.seq[self.guess_position_history[self.finished_steps]], " at ", self.guess_position_history[self.finished_steps])

        #revert the mask
        self.mask = np.copy(self.mask_history[self.finished_steps - 1,:,:])
        self.seq = np.copy(self.seq_history[self.finished_steps - 1,:])


        #delete the last entry of the histories
        self.seq_history = np.delete(self.seq_history, self.finished_steps, 0)
        self.mask_history = np.delete(self.mask_history, self.finished_steps, 0)
        self.guess_position_history = np.delete(self.guess_position_history, self.finished_steps, 0)

        #update the finished steps
        self.finished_steps = self.finished_steps - 1

        #values after reversion
        # print ("After reversion: ")
        # print ("seq ", self.seq)
        # print("mask ", self.mask)
        # print("Seq history: ", self.seq_history)
        # print("Mask history: ", self.mask_history)

        #return the reverted sequence
        return True


    #a function that makes the next speculative step in the sequence by using the first free value allowed by the mask
    def make_next_guess(self, slot):
        #check if there are no allowed slots in the mask:
        if np.sum(self.mask[slot,:]) == 0:
            # print("No allowed slots in the mask at position: ", slot)
            # print(self.mask[slot,:])
            
            return False
        
        #otherwise, make the guess
        # print("Making a guess at slot: ", slot, " with mask ", self.mask[slot,:], " we guessed ", np.where(self.mask[slot,:] == 1)[0][0])

        self.seq[slot] = np.where(self.mask[slot,:] == 1)[0][0] #this is the first allowed value in the mask

        return True


    #the main function that keeps guessing and logically concluding, and where neccessary, backtracking and resetting the history

    def guess_and_conclude(self):

        print("Starting guess and conclude")

        first_open_slot = np.where(self.seq == -1)[0][0]

        active_slot = first_open_slot

        iterated_steps = 0

        #main loop
        while True:

            iterated_steps += 1

            # if iterated_steps > 1000:
            #      print("Too many iterations")
            #      self.print_last_sequence_and_mask_before_guess()
            #      return False

            #check if we are done
            if np.size(np.where(self.seq == -1)) == 0:
                return True

            # print("Active slot: ", active_slot)

            #make a guess
            if(self.make_next_guess(active_slot)):

                #check if we are consistent
                result_of_logical_updates = self.logical_update_until_convergence(active_slot)

                if result_of_logical_updates == 1:
                    #update the histories and increase the finished steps
                    self.seq_history = np.vstack((self.seq_history, self.seq))
                    self.mask_history = np.vstack((self.mask_history, np.reshape(self.mask, (1, self.mask.shape[0], self.mask.shape[1] ))) )
                    self.guess_position_history = np.hstack((self.guess_position_history, active_slot))

                    self.finished_steps += 1
                    active_slot = np.where(self.seq == -1)[0][0] #the next open slot

                    print("Finished step: ", self.finished_steps, " with sequence ", self.seq)
                    # self.print_seq_without_trailing_empty()


                elif result_of_logical_updates == 2:
                    print("Delayed logically induced argument exceeded maximum allowed space! Printing final status.")
                    self.print_last_sequence_and_mask_before_guess()
                    
                    return False #we ran out of length

                else:
                    #we have a contradiction, so we need to make a new guess
                    

                    #update the mask history for the last guessed value to be 0
                    self.mask_history[self.finished_steps, active_slot, self.seq[active_slot]] = 0

                    self.mask = np.copy(self.mask_history[self.finished_steps,:,:]) #reset the mask to the state at the start of this attempt

                    # print("Contradiction at step: ", self.finished_steps, " with sequence ", self.seq)



                    self.seq = np.copy(self.seq_history[self.finished_steps,:]) #reset the sequence to the state at the start of this attempt

                    # print("Reverting to saved history from start of step : ", self.finished_steps, " with sequence ", self.seq)

            else:
                #we ran out of guesses, so we need to revert a step and try again from an earlier point
                #revert a step
                if(self.revert_a_step()):
                    print("~ Reverted a step! Now at step: ", self.finished_steps, " again ~")

                    active_slot = np.where(self.seq == -1)[0][0] #the next open slot
                else:
                    return False





    def print_last_sequence_and_mask_before_guess(self):
        print("Last sequence before guess: ")
        self.print_seq_without_trailing_empty()
        #print("mask: ", self.mask_history[self.finished_steps,:,:])

        validity = check_advanced_van_eck_property(self.seq_history[self.finished_steps,:], True)
        print("Validity: ", validity)



    def print_status(self):
        print("finished steps: ", self.finished_steps)
        print("last updated position: ", self.guess_position_history[self.finished_steps])
        print("last logical update: ", self.logical_update_position)
        print("last sequence before: ", self.seq)
        print("mask: ", self.mask)







test_seq_1 = masked_van_eck(1000, 100, np.array([0,0,1]))

#test_seq_1.print_status()



#print(test_seq_1.logical_update_until_convergence() )

#test_seq_1.print_status()

test_seq_1.guess_and_conclude()

#test_seq_1.print_status()

