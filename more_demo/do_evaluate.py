"""Call the Python API flowbench.evals.evaluate(...) of FlowBench.
It's less customizable than directly calling flowbench.attacks.attack(args, ...) but 
"""


from my_new_model import * # registers model to PTLFlow

from flowbench.evals import evaluate

if __name__ == '__main__':
    import ptlflow
    
    # Check that the model is registered, i.e., known to ptlflow
    model = ptlflow.get_model('my_very_own_raft')

    # Do evaluations    
    evaluate(
        model_name='my_very_own_raft',
        dataset='sintel-final',
        threat_model='PCFA',
        retrieve_existing=False,
    )
