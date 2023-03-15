def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'double':
        from .doubleNetG_model import SingleModel
        model = SingleModel()
    elif opt.model == 'separate':
        from .separateNetG_model import SingleModel
        model = SingleModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
