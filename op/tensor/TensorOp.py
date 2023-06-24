import torch


class TensorOp:
    def __init__(self):
        pass

    @staticmethod
    def flatten(tensor):
        """flatten a arbitrary tensor to a single axe scalar"""
        t = tensor.reshape(1, -1)
        t = t.squeeze()
        return t

    @staticmethod
    def tpad(tensor, dim, n, fillvalue, device=torch.device("cuda:0")):
        """
        pad a tensor along a dimension with fillvalue
        :param tensor: source tensor
        :param dim: along which dimension to perform padding
        :param n: pad length
        :param fillvalue: pad value
        :return: padded tensor
        """
        tensor = tensor.to(device)

        nlist = [1 if i != dim else n for i in range(len(tensor.shape))]
        tmp = torch.index_select(tensor, dim, torch.tensor([0]).to(device))
        tmp = tmp.fill_(fillvalue)
        tmp = tmp.repeat(nlist)
        tmp = tmp.to(device)
        r = torch.cat((tensor, tmp), dim)
        return r

    @staticmethod
    def bool_narrow2D(mask, dim=0, crit="any"):
        """
        narrow a 2D mask tensor[m, n] to [m, 1] if dim=0 ; to [1,n] if dim=1
        :param mask: mask tensor
        :param dim: along which dim to narrow
        :return: narrowed mask tensor
        """
        if crit == "any":
            return mask.any(dim)
        else:
            return mask.all(dim)
        # if dim == 0:
        #     for i in range(mask.shape[0]):
        #         row = mask[i]
        #         if row.all():
        #             mask[i][0] = True
        #         else:
        #             mask[i][0] = False
        #     r = torch.narrow(mask, 1, 0, 1)
        #     return r
        # elif dim == 1:
        #     for i in range(mask.shape[1]):
        #         column = mask[:,i]
        #         if column.all():
        #             mask[0][i] = True
        #         else:
        #             mask[0][i] = False
        #     r = torch.narrow(mask,0,0,1)
        #     return r
        # else:
        #     raise Exception

    @staticmethod
    def bool_narrow3D(mask, dim=-1, crit="any"):
        """
        now only support dim = -1 for 3D mask tensor
        :param mask:
        :param dim:
        :return:
        """
        if crit == "any":
            return mask.any(dim)
        else:
            return mask.all(dim)
        #
        # for i in range(mask.shape[dim]):
        #     axe = torch.index_select(mask, dim, torch.tensor([i])).squeeze()
        #     for j in range(axe.shape[0]):
        #         if axe[j].all():
        #             mask[j][i][0] = True
        #         else:
        #             mask[j][i][0] = False
        # r = torch.narrow(mask, -1, 0, 1).squeeze()
        # return r
